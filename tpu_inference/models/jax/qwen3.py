# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from transformers import Qwen3Config
from vllm.config import VllmConfig
from vllm.transformers_utils.config import set_default_rope_theta

from tpu_inference import envs, utils
from tpu_inference.distributed.jax_parallel_state import get_pp_group
from tpu_inference.kernels.collectives.fused_all_reduce_matmul import \
    fused_all_reduce_matmul
from tpu_inference.kernels.fused_rmsnorm_quant.fused_rmsnorm_fp8 import (
    fused_rmsnorm_fp8_quant,
)
from tpu_inference.kernels.swiglu.fused_swiglu_pallas import (
    fused_swiglu_pallas,
)
try:
    from tpu_inference.kernels.experimental.brpa_rope.pallas_rmsnorm import (
        pallas_2d_rmsnorm,
    )
except ImportError:
    try:
        from google3.experimental.users.fangfangz.kernels.brpa_rope.pallas_rmsnorm import (
            pallas_2d_rmsnorm,
        )
    except ImportError:
        pallas_2d_rmsnorm = None
from tpu_inference.layers.common.attention_interface import attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.quantization import quantize_kv
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.embed import JaxEmbed
from tpu_inference.layers.jax.layers import FlaxUtils
from tpu_inference.layers.jax.linear import JaxEinsum, JaxLinear, JaxLmHead
from tpu_inference.layers.jax.norm import JaxRmsNorm
from tpu_inference.layers.jax.pp_utils import PPMissingLayer, make_layers
from tpu_inference.layers.jax.rope_interface import (apply_rope,
                                                     get_rope_scaling,
                                                     get_rope_theta)
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig

try:
    from qwix._src.core.qarray import QArray
    from qwix._src.core import qarray as _qwix_qarray
    from qwix._src.providers import ptq as _qwix_ptq

    if not getattr(_qwix_qarray.quantize, "_is_qarray_safe", False):
        _orig_quantize = _qwix_qarray.quantize
        def _safe_quantize(array, how):
            if hasattr(array, "qvalue") or isinstance(array, _qwix_qarray.QArray):
                return array
            return _orig_quantize(array, how)
        _safe_quantize._is_qarray_safe = True
        _qwix_qarray.quantize = _safe_quantize

    if hasattr(_qwix_ptq, "quantize_act") and not getattr(_qwix_ptq.quantize_act, "_is_qarray_safe", False):
        _orig_quantize_act = _qwix_ptq.quantize_act
        def _safe_quantize_act(array, how, rule, act_name, **kwargs):
            if hasattr(array, "qvalue") or isinstance(array, _qwix_qarray.QArray):
                return array
            return _orig_quantize_act(array, how, rule, act_name, **kwargs)
        _safe_quantize_act._is_qarray_safe = True
        _qwix_ptq.quantize_act = _safe_quantize_act
except ImportError:
    QArray = None
    _qwix_ptq = None
ptq = _qwix_ptq
_QWIX_TYPES = tuple(c for c in (getattr(_qwix_ptq, "WithAux", None), QArray) if c is not None)
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.qwen2 import Qwen2DecoderLayer, Qwen2Model
from tpu_inference.models.jax.utils.qwix.qwix_utils import (
    manually_quantize_qwix_weight,
)
from tpu_inference.models.jax.utils.weight_utils import LoadableWithIterator

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()
modeling_flax_utils = FlaxUtils()


def _apply_fused_rmsnorm_fp8(
    x: jax.Array,
    gamma: jax.Array,
    mesh: Mesh | None = None,
    residual: jax.Array | None = None,
    return_residual: bool = False,
    eps: float = 1e-6,
    block_k: int | None = None,
) -> tuple[jax.Array, ...] | None:
    """Applies fused RMSNorm + FP8 dynamic quantization if shape is compatible."""
    m = x.shape[0]
    target_block_m = envs.FUSED_RMSNORM_BLOCK_M or 512
    block_m = target_block_m
    for candidate in [target_block_m, 256, 128, 64, 32, 16]:
        if m % candidate == 0 and candidate <= m:
            block_m = candidate
            break
    if m % block_m != 0:
        return None

    def _call_kernel(x_loc, gamma_loc, res_loc=None):
        return fused_rmsnorm_fp8_quant(
            x_loc,
            gamma_loc,
            residual=res_loc,
            block_m=block_m,
            block_k=block_k,
            eps=eps,
            return_residual=return_residual,
        )

    if mesh is not None and len(mesh.devices.shape) > 0 and mesh.devices.size > 1:
        if residual is not None:
            return jax.shard_map(
                _call_kernel,
                mesh=mesh,
                in_specs=(jax.sharding.PartitionSpec(), jax.sharding.PartitionSpec(), jax.sharding.PartitionSpec()),
                out_specs=(
                    (jax.sharding.PartitionSpec(), jax.sharding.PartitionSpec(), jax.sharding.PartitionSpec())
                    if return_residual
                    else (jax.sharding.PartitionSpec(), jax.sharding.PartitionSpec())
                ),
                check_vma=False,
            )(x, gamma, residual)
        else:
            return jax.shard_map(
                _call_kernel,
                mesh=mesh,
                in_specs=(jax.sharding.PartitionSpec(), jax.sharding.PartitionSpec()),
                out_specs=(jax.sharding.PartitionSpec(), jax.sharding.PartitionSpec()),
                check_vma=False,
            )(x, gamma)

    return _call_kernel(x, gamma, residual)


def _apply_pallas_2d_rmsnorm(
    q: jax.Array,
    gamma: jax.Array,
    mesh: Mesh | None = None,
    eps: float = 1e-6,
) -> jax.Array:
    """Applies dedicated 2D Pallas RMSNorm kernel to head-major Q tensor.

    Args:
        q: [num_kv_heads, total_tokens, g, head_dim] sharded along num_kv_heads on 'model'.
        gamma: [head_dim]
        mesh: JAX device mesh.
        eps: RMSNorm epsilon.

    Returns:
        Normalized Q tensor of shape [num_kv_heads, total_tokens, g, head_dim].
    """
    if pallas_2d_rmsnorm is None:
        n_kv, total_tokens, g, head_dim = q.shape
        q_3d = q.reshape(n_kv, total_tokens * g, head_dim)
        var = jnp.mean(jnp.square(q_3d.astype(jnp.float32)), axis=-1, keepdims=True)
        q_norm = (q_3d.astype(jnp.float32) * jax.lax.rsqrt(var + eps)) * gamma.astype(jnp.float32)
        return q_norm.astype(q.dtype).reshape(n_kv, total_tokens, g, head_dim)

    def _call_norm(q_loc, gamma_loc):
        return pallas_2d_rmsnorm(q_loc, gamma_loc, eps=eps)

    if (
        mesh is not None
        and len(mesh.devices.shape) > 0
        and mesh.devices.size > 1
        and "model" in mesh.axis_names
        and mesh.shape["model"] > 1
    ):
        in_specs = (
            jax.sharding.PartitionSpec("model", None, None, None),
            jax.sharding.PartitionSpec(),
        )
        out_specs = jax.sharding.PartitionSpec("model", None, None, None)
        return jax.shard_map(
            _call_norm,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )(q, gamma)
    else:
        return _call_norm(q, gamma)


def _apply_fused_swiglu(
    lhs: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    mesh: Mesh | None = None,
    tile_m: int | None = None,
    tile_n: int | None = None,
    tile_k: int | None = None,
    pipeline_mode: str = "grid",
    quant_mode: str = "channelwise_separate_pallas",
    quant_out: bool = True,
    subchannel_k: int = 512,
    lhs_scale: jax.Array | None = None,
    w_gate_scale: jax.Array | None = None,
    w_up_scale: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array] | jax.Array | None:
    """Dispatches Fused SwiGLU Pallas TPU kernel with sharding and candidate tiling."""
    if lhs_scale is None:
        lhs_q, lhs_scale = _quantize_to_fp8_with_scale(lhs, channelwise_axis=-1, target_qtype=jnp.float8_e4m3fn)
    else:
        lhs_q = lhs.qvalue if hasattr(lhs, "qvalue") else (lhs.array.qvalue if hasattr(lhs, "array") and hasattr(lhs.array, "qvalue") else lhs)

    if w_gate_scale is None:
        w_gate_q, w_gate_scale = _quantize_to_fp8_with_scale(w_gate, channelwise_axis=0, target_qtype=jnp.float8_e4m3fn)
    else:
        w_gate_q = w_gate.qvalue if hasattr(w_gate, "qvalue") else (w_gate.array.qvalue if hasattr(w_gate, "array") and hasattr(w_gate.array, "qvalue") else w_gate)

    if w_up_scale is None:
        w_up_q, w_up_scale = _quantize_to_fp8_with_scale(w_up, channelwise_axis=0, target_qtype=jnp.float8_e4m3fn)
    else:
        w_up_q = w_up.qvalue if hasattr(w_up, "qvalue") else (w_up.array.qvalue if hasattr(w_up, "array") and hasattr(w_up.array, "qvalue") else w_up)

    if lhs_scale is None:
        lhs_scale = jnp.ones((lhs_q.shape[0], 1), dtype=jnp.bfloat16)
    elif lhs_scale.ndim == 1:
        lhs_scale = lhs_scale[:, None]

    if w_gate_scale is None:
        w_gate_scale = jnp.ones((1, w_gate_q.shape[1]), dtype=jnp.bfloat16)
    elif w_gate_scale.ndim == 1:
        w_gate_scale = w_gate_scale[None, :]

    if w_up_scale is None:
        w_up_scale = jnp.ones((1, w_up_q.shape[1]), dtype=jnp.bfloat16)
    elif w_up_scale.ndim == 1:
        w_up_scale = w_up_scale[None, :]

    m, k = lhs_q.shape
    num_devices = mesh.shape["model"] if (mesh is not None and "model" in mesh.axis_names) else 1
    n_local = w_gate_q.shape[1] // num_devices if mesh is not None else w_gate_q.shape[1]

    # Candidate search for tile_m
    target_m = tile_m or envs.FUSED_SWIGLU_BLOCK_M or 2048
    chosen_m = None
    for cand in [target_m, 1024, 512, 256, 128, 64, 32, 16]:
        if m % cand == 0 and cand <= m:
            chosen_m = cand
            break
    if chosen_m is None:
        return None

    # Candidate search for tile_n
    target_n = tile_n or envs.FUSED_SWIGLU_BLOCK_N or 512
    chosen_n = None
    for cand in [target_n, 256, 128, 64]:
        if n_local % cand == 0 and cand <= n_local:
            chosen_n = cand
            break
    if chosen_n is None:
        return None

    chosen_k = tile_k or envs.FUSED_SWIGLU_BLOCK_K or k

    def _call_kernel(lhs_q_loc, lhs_s_loc, wg_q_loc, wg_s_loc, wu_q_loc, wu_s_loc):
        return fused_swiglu_pallas(
            lhs_q=lhs_q_loc,
            lhs_scale=lhs_s_loc,
            w_gate_q=wg_q_loc,
            w_gate_scale=wg_s_loc,
            w_up_q=wu_q_loc,
            w_up_scale=wu_s_loc,
            tile_m=chosen_m,
            tile_n=chosen_n,
            tile_k=chosen_k,
            quant_out=quant_out,
            quant_mode=quant_mode,
            subchannel_k=subchannel_k,
            pipeline_mode=pipeline_mode,
        )

    if mesh is not None and len(mesh.devices.shape) > 0 and mesh.devices.size > 1 and "model" in mesh.axis_names and mesh.shape["model"] > 1:
        in_specs = (
            jax.sharding.PartitionSpec(),  # lhs_q replicated
            jax.sharding.PartitionSpec(),  # lhs_scale replicated
            jax.sharding.PartitionSpec(None, "model"),  # w_gate_q sharded
            jax.sharding.PartitionSpec(None, "model"),  # w_gate_scale sharded
            jax.sharding.PartitionSpec(None, "model"),  # w_up_q sharded
            jax.sharding.PartitionSpec(None, "model"),  # w_up_scale sharded
        )
        out_specs = (
            (jax.sharding.PartitionSpec(None, "model"), jax.sharding.PartitionSpec(None, "model"))
            if quant_out else jax.sharding.PartitionSpec(None, "model")
        )
        return jax.shard_map(
            _call_kernel,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )(lhs_q, lhs_scale, w_gate_q, w_gate_scale, w_up_q, w_up_scale)

    return _call_kernel(lhs_q, lhs_scale, w_gate_q, w_gate_scale, w_up_q, w_up_scale)


def _is_fp8_qwix_enabled(additional_config: dict | None, prefix: str) -> bool:
    """Checks if Qwix rules specify FP8 quantization for the given module prefix."""
    if not additional_config or not isinstance(additional_config, dict):
        return False
    quant_cfg = additional_config.get("quantization")
    if not quant_cfg or not isinstance(quant_cfg, dict):
        return False
    qwix_cfg = quant_cfg.get("qwix")
    if not qwix_cfg or not isinstance(qwix_cfg, dict):
        return False
    rules = qwix_cfg.get("rules", [])
    for rule in rules:
        pattern = rule.get("module_path", ".*")
        if re.search(pattern, prefix):
            w_qtype = str(rule.get("weight_qtype", "")).lower()
            a_qtype = str(rule.get("act_qtype", "")).lower()
            if "float8" in w_qtype or "fp8" in w_qtype or "float8" in a_qtype or "fp8" in a_qtype:
                return True
    return False


def _quantize_to_fp8_with_scale(
    arr: jax.Array,
    channelwise_axis: int | None = None,
    target_qtype: jnp.dtype = jnp.float8_e4m3fn,
) -> tuple[jax.Array, jax.Array | None]:
    """Quantizes an array to target FP8 dtype and computes scaling factors.

    If the array is already a QArray or FP8, extracts qvalue and scale.
    Otherwise, applies dynamic absmax quantization.
    """
    if hasattr(arr, "qvalue"):
        qval = arr.qvalue
        scale = getattr(arr, "scale", None)
        return qval, scale
    if hasattr(arr, "array") and hasattr(arr.array, "qvalue"):
        qval = arr.array.qvalue
        scale = getattr(arr.array, "scale", None)
        return qval, scale
    if arr.dtype == target_qtype:
        return arr, None

    if channelwise_axis is not None:
        arr_max = jnp.max(jnp.abs(arr), axis=channelwise_axis, keepdims=True)
    else:
        arr_max = jnp.max(jnp.abs(arr), keepdims=True)

    scale = jnp.maximum(arr_max / 448.0, 1e-12).astype(arr.dtype)
    qval = jnp.clip(arr / scale, -448.0, 448.0).astype(target_qtype)
    return qval, scale


def _get_qwix_fp8_weight(
    layer: Any,
    name: str,
    additional_config: dict | None,
    channelwise_axis: int = 0,
) -> tuple[jax.Array, jax.Array | None]:
    """Retrieves or quantizes weight for Qwix FP8 flow.

    If the weight parameter is already a QArray / WithAux (quantized by Qwix),
    extracts the (qvalue, scale) directly.
    If not yet quantized and Qwix FP8 is active, manually quantizes the weight
    using Qwix (manually_quantize_qwix_weight) and updates layer.weight.
    """
    weight_param = getattr(layer, "kernel", getattr(layer, "weight", None))
    if weight_param is None:
        return _quantize_to_fp8_with_scale(layer, channelwise_axis=channelwise_axis, target_qtype=jnp.float8_e4m3fn)

    weight = getattr(weight_param, "value", weight_param)
    if hasattr(weight, "qvalue"):
        return weight.qvalue, getattr(weight, "scale", None)
    if hasattr(weight, "array") and hasattr(weight.array, "qvalue"):
        return weight.array.qvalue, getattr(weight.array, "scale", None)
    if hasattr(weight, "dtype") and weight.dtype == jnp.float8_e4m3fn:
        return weight, None

    return _quantize_to_fp8_with_scale(weight, channelwise_axis=channelwise_axis, target_qtype=jnp.float8_e4m3fn)



class Qwen3Attention(JaxModule):

    def __init__(self,
                 config: Qwen3Config,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 quant_config: VllmQuantConfig,
                 additional_config: dict | None = None,
                 prefix: str = ""):
        set_default_rope_theta(config, default_theta=1000000)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.rope_theta = get_rope_theta(config, default=1000000.0)
        self.rope_scaling = get_rope_scaling(config)
        self.rms_norm_eps = config.rms_norm_eps
        self.additional_config = additional_config
        self.prefix = prefix

        self.head_dim_original = getattr(config, "head_dim",
                                         self.hidden_size // self.num_heads)
        self.head_dim = utils.get_padded_head_dim(self.head_dim_original)

        sharding_size = mesh.shape["model"]
        self.num_heads = utils.get_padded_num_heads(self.num_heads,
                                                    sharding_size)
        self.num_kv_heads = utils.get_padded_num_heads(self.num_kv_heads,
                                                       sharding_size)

        self.mesh = mesh
        self.additional_config = additional_config

        if envs.USE_HEAD_MAJOR_Q_JOINT_KV_RPA or envs.USE_KV_HEAD_MAJOR_IN_KERNEL_ROPE_RPA:
            rhs_str = "KDN"
            num_q_heads_per_kv = self.num_heads // self.num_kv_heads
            q_proj_sharding = ("model", None, None)
            kernel_shape = (
                self.num_kv_heads,
                self.hidden_size,
                num_q_heads_per_kv * self.head_dim,
            )
            einsum_str = f"KTD,{rhs_str}->KTN"
        elif envs.LAYOUT_Q_PROJ_AS_NDH:
            rhs_str = "NDH"
            q_proj_sharding = ("model", None, None)
            kernel_shape = (self.num_heads, self.hidden_size, self.head_dim)
            einsum_str = f"TD,{rhs_str}->TNH"
        else:
            rhs_str = "DNH"
            q_proj_sharding = (None, "model", None)
            kernel_shape = (self.hidden_size, self.num_heads, self.head_dim)
            einsum_str = f"TD,{rhs_str}->TNH"

        logger.info_once(
            f"Running with attention Q-Projection laid out as {rhs_str}")

        self.q_proj = JaxEinsum(
            einsum_str,
            kernel_shape,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, q_proj_sharding),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".q_proj",
        )
        self.q_norm = JaxRmsNorm(
            self.head_dim,
            epsilon=self.rms_norm_eps,
            dtype=dtype,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".q_norm",
        )
        if envs.USE_HEAD_MAJOR_Q_JOINT_KV_RPA:
            self.kv_proj = JaxEinsum(
                "TD,DH->TH",
                (self.hidden_size, 2 * self.num_kv_heads * self.head_dim),
                dtype=dtype,
                param_dtype=dtype,
                kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + ".kv_proj",
            )
            self.k_norm = JaxRmsNorm(
                self.head_dim,
                epsilon=self.rms_norm_eps,
                dtype=dtype,
                param_dtype=dtype,
                scale_init=nnx.with_partitioning(init_fn, (None, )),
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + ".k_norm",
            )
        else:
            self.k_proj = JaxEinsum(
                "TD,DKH->TKH",
                (self.hidden_size, self.num_kv_heads, self.head_dim),
                dtype=dtype,
                param_dtype=dtype,
                kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + ".k_proj",
            )
            self.k_norm = JaxRmsNorm(
                self.head_dim,
                epsilon=self.rms_norm_eps,
                dtype=dtype,
                param_dtype=dtype,
                scale_init=nnx.with_partitioning(init_fn, (None, )),
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + ".k_norm",
            )
            self.v_proj = JaxEinsum(
                "TD,DKH->TKH",
                (self.hidden_size, self.num_kv_heads, self.head_dim),
                dtype=dtype,
                param_dtype=dtype,
                kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + ".v_proj",
            )
        self.o_proj = JaxEinsum(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, self.hidden_size),
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".o_proj",
        )

        self._q_scale = 1.0
        self._k_scale = 1.0
        self._v_scale = 1.0
        self.use_in_kernel_rope = (
            getattr(config, "use_in_kernel_rope", False)
            or envs.USE_STRIDED_IN_KERNEL_ROPE_RPA
            or envs.USE_KV_HEAD_MAJOR_IN_KERNEL_ROPE_RPA
            or envs.USE_HEAD_MAJOR_Q_JOINT_KV_RPA
        )
        self.kv_cache_quantized_dtype = None
        if kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = utils.get_jax_dtype_from_str_dtype(
                kv_cache_dtype)

    def __call__(
        self,
        kv_cache: Optional[jax.Array],
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        md = attention_metadata
        use_in_kernel_rope = (
            self.use_in_kernel_rope
            or envs.USE_STRIDED_IN_KERNEL_ROPE_RPA
            or envs.USE_KV_HEAD_MAJOR_IN_KERNEL_ROPE_RPA
            or envs.USE_HEAD_MAJOR_Q_JOINT_KV_RPA
        )

        # q: (T, N, H) or (K, T, G, H)
        if envs.USE_KV_HEAD_MAJOR_IN_KERNEL_ROPE_RPA or envs.USE_HEAD_MAJOR_Q_JOINT_KV_RPA:
            if hasattr(x, "qvalue"):
                x_qval = jnp.broadcast_to(
                    x.qvalue[None, :, :], (self.num_kv_heads, x.shape[0], self.hidden_size)
                )
                x_scale = (
                    jnp.broadcast_to(x.scale[None, :, :], (self.num_kv_heads, x.shape[0], 1))
                    if x.scale is not None else None
                )
                x_q = QArray(qvalue=x_qval, scale=x_scale, qtype=x.qtype)
            elif hasattr(x, "array") and hasattr(x.array, "qvalue"):
                x_qval = jnp.broadcast_to(
                    x.array.qvalue[None, :, :], (self.num_kv_heads, x.shape[0], self.hidden_size)
                )
                x_scale = (
                    jnp.broadcast_to(x.array.scale[None, :, :], (self.num_kv_heads, x.shape[0], 1))
                    if getattr(x.array, "scale", None) is not None else None
                )
                x_q = QArray(qvalue=x_qval, scale=x_scale, qtype=getattr(x.array, "qtype", None))
            else:
                x_q = jnp.broadcast_to(
                    x[None, :, :], (self.num_kv_heads, x.shape[0], self.hidden_size)
                )
            q = self.q_proj(x_q)
            n_kv, total_tokens, gd = q.shape
            g = gd // self.head_dim
            q_4d = q.reshape(n_kv, total_tokens, g, self.head_dim)
            gamma_q = getattr(self.q_norm.weight, "value", self.q_norm.weight)
            if envs.USE_PALLAS_2D_RMSNORM:
                q = _apply_pallas_2d_rmsnorm(
                    q_4d,
                    gamma_q,
                    mesh=self.mesh,
                    eps=self.rms_norm_eps,
                )
            else:
                q_3d = q_4d.reshape(n_kv, total_tokens * g, self.head_dim)
                q_norm_3d = self.q_norm(q_3d)
                q = q_norm_3d.reshape(n_kv, total_tokens, g, self.head_dim)
        else:
            q = self.q_proj(x)
            q = self.q_norm(q)

        # k, v projections
        if envs.USE_HEAD_MAJOR_Q_JOINT_KV_RPA:
            kv = self.kv_proj(x)
            kv = kv.reshape(x.shape[0], 2, self.num_kv_heads, self.head_dim)
            k_raw = kv[:, 0, :, :]
            v = kv[:, 1, :, :]
            k_shape = k_raw.shape
            k_norm_flat = self.k_norm(k_raw.reshape((-1, self.head_dim)))
            k = k_norm_flat.reshape(k_shape)
        else:
            # k: (T, K, H)
            k = self.k_proj(x)
            k = self.k_norm(k)

            # v: (T, K, H)
            v = self.v_proj(x)

        q_scale = k_scale = v_scale = None
        if not use_in_kernel_rope:
            q = apply_rope(q, md.input_positions, self.head_dim_original,
                           self.rope_theta, self.rope_scaling)
            k = apply_rope(k, md.input_positions, self.head_dim_original,
                           self.rope_theta, self.rope_scaling)
        if self.kv_cache_quantized_dtype:
            k_scale = self._k_scale
            v_scale = self._v_scale
            k, v = quantize_kv(self.kv_cache_quantized_dtype, k, v, k_scale,
                               v_scale)

        new_kv_cache, outputs = attention(
            kv_cache,
            q,
            k,
            v,
            attention_metadata,
            self.mesh,
            self.head_dim_original,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            use_in_kernel_rope=use_in_kernel_rope,
            rope_theta=self.rope_theta,
        )
        if envs.USE_KV_HEAD_MAJOR_IN_KERNEL_ROPE_RPA or envs.USE_HEAD_MAJOR_Q_JOINT_KV_RPA:
            outputs = outputs.swapaxes(0, 1).reshape(
                x.shape[0], self.num_heads, self.head_dim
            )
        # (T, D)
        if envs.USE_FUSED_ALL_REDUCE_MATMUL:
            if (
                getattr(self, "disable_quant_stats_update", False)
                
                
            ):
                return new_kv_cache, self.o_proj(outputs)
            outputs_2d = outputs.reshape((outputs.shape[0], -1))
            w_o_q, w_o_scale = _get_qwix_fp8_weight(
                self.o_proj, f"{self.prefix}.o_proj", self.additional_config, channelwise_axis=0
            )
            if hasattr(outputs_2d, "qvalue"):
                x_in = outputs_2d.qvalue
            elif hasattr(outputs_2d, "array") and hasattr(outputs_2d.array, "qvalue"):
                x_in = outputs_2d.array.qvalue
            elif (w_o_q.dtype == jnp.float8_e4m3fn or
                  _is_fp8_qwix_enabled(self.additional_config, f"{self.prefix}.o_proj") or
                  (hasattr(self.o_proj, "quant_config") and self.o_proj.quant_config is not None)):
                x_in, _ = _quantize_to_fp8_with_scale(outputs_2d, channelwise_axis=-1, target_qtype=jnp.float8_e4m3fn)
            else:
                x_in = outputs_2d.astype(w_o_q.dtype)
            w_o = w_o_q.reshape((-1, self.hidden_size))
            num_devices = self.mesh.shape["model"]
            k_local = x_in.shape[1] // num_devices
            block_m = envs.FUSED_AR_ATTN_BLOCK_M or envs.FUSED_AR_BLOCK_M or min(outputs_2d.shape[0], 2048)
            block_n = envs.FUSED_AR_ATTN_BLOCK_N or envs.FUSED_AR_BLOCK_N or min(self.hidden_size, 1024)
            block_k = envs.FUSED_AR_ATTN_BLOCK_K or envs.FUSED_AR_BLOCK_K or min(k_local, 2048)
            o = fused_all_reduce_matmul(
                x_in,
                w_o,
                mesh=self.mesh,
                axis_name="model",
                out_dtype=outputs.dtype,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                k_local=k_local,
                pipeline_mode=envs.FUSED_AR_PIPELINE_MODE or "5stage",
            )
        else:
            o = self.o_proj(outputs)
        return new_kv_cache, o


class Qwen3MLP(JaxModule):

    def __init__(self,
                 config: Qwen3Config,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 quant_config: VllmQuantConfig,
                 additional_config: dict | None = None,
                 prefix: str = ""):
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        act = config.hidden_act

        self.mesh = mesh
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.additional_config = additional_config
        self.prefix = prefix

        self.gate_proj = JaxLinear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".gate_proj",
        )
        self.up_proj = JaxLinear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".up_proj",
        )
        self.down_proj = JaxLinear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".down_proj",
        )
        self.act_fn = modeling_flax_utils.ACT2FN[act]

    def __call__(self, x: jax.Array) -> jax.Array:
        if (
            getattr(self, "disable_quant_stats_update", False)
            
            
        ):
            gate = self.act_fn(self.gate_proj(x))
            up = self.up_proj(x)
            fuse = gate * up
            return self.down_proj(fuse)
        if envs.USE_FUSED_SWIGLU:
            x_2d = x.reshape((-1, x.shape[-1])) if x.ndim > 2 else x
            w_gate_q, w_gate_scale = _get_qwix_fp8_weight(
                self.gate_proj, f"{self.prefix}.gate_proj", self.additional_config, channelwise_axis=0
            )
            w_up_q, w_up_scale = _get_qwix_fp8_weight(
                self.up_proj, f"{self.prefix}.up_proj", self.additional_config, channelwise_axis=0
            )
            swiglu_res = _apply_fused_swiglu(
                lhs=x_2d,
                w_gate=w_gate_q,
                w_up=w_up_q,
                mesh=self.mesh,
                tile_m=envs.FUSED_SWIGLU_BLOCK_M,
                tile_n=envs.FUSED_SWIGLU_BLOCK_N,
                tile_k=envs.FUSED_SWIGLU_BLOCK_K,
                pipeline_mode=envs.FUSED_SWIGLU_PIPELINE_MODE or "grid",
                quant_mode=envs.FUSED_SWIGLU_QUANT_MODE or "channelwise_separate_pallas",
                quant_out=True,
                subchannel_k=envs.FUSED_SWIGLU_SUBCHANNEL_K or 512,
                w_gate_scale=w_gate_scale,
                w_up_scale=w_up_scale,
            )
            if swiglu_res is not None:
                fuse_q, fuse_scale = swiglu_res
                if QArray is not None:
                    fuse = QArray(qvalue=fuse_q, scale=fuse_scale, qtype=jnp.float8_e4m3fn)
                    if x.ndim > 2:
                        fuse = fuse.reshape((*x.shape[:-1], self.intermediate_size))
                else:
                    fuse = fuse_q.astype(jnp.bfloat16) * fuse_scale
                    if x.ndim > 2:
                        fuse = fuse.reshape((*x.shape[:-1], self.intermediate_size))
            else:
                gate = self.act_fn(self.gate_proj(x))
                up = self.up_proj(x)
                fuse = gate * up
        else:
            gate = self.act_fn(self.gate_proj(x))
            up = self.up_proj(x)
            fuse = gate * up
        if envs.USE_FUSED_ALL_REDUCE_MATMUL:
            fuse_2d = fuse.reshape((-1, fuse.shape[-1])) if fuse.ndim > 2 else fuse
            w_down_q, w_down_scale = _get_qwix_fp8_weight(
                self.down_proj, f"{self.prefix}.down_proj", self.additional_config, channelwise_axis=0
            )
            if hasattr(fuse_2d, "qvalue"):
                fuse_in = fuse_2d.qvalue
            elif hasattr(fuse_2d, "array") and hasattr(fuse_2d.array, "qvalue"):
                fuse_in = fuse_2d.array.qvalue
            elif (w_down_q.dtype == jnp.float8_e4m3fn or
                  _is_fp8_qwix_enabled(self.additional_config, f"{self.prefix}.down_proj") or
                  (hasattr(self.down_proj, "quant_config") and self.down_proj.quant_config is not None)):
                fuse_in, _ = _quantize_to_fp8_with_scale(fuse_2d, channelwise_axis=-1, target_qtype=jnp.float8_e4m3fn)
            else:
                fuse_in = fuse_2d.astype(w_down_q.dtype)
            w_down = w_down_q.reshape((-1, self.hidden_size))
            num_devices = self.mesh.shape["model"]
            k_local = fuse_in.shape[-1] // num_devices
            block_m = envs.FUSED_AR_MLP_BLOCK_M or envs.FUSED_AR_BLOCK_M or min(fuse_2d.shape[0], 2048)
            block_n = envs.FUSED_AR_MLP_BLOCK_N or envs.FUSED_AR_BLOCK_N or min(self.hidden_size, 1024)
            block_k = envs.FUSED_AR_MLP_BLOCK_K or envs.FUSED_AR_BLOCK_K or min(k_local, 2048)
            out = fused_all_reduce_matmul(
                fuse_in,
                w_down,
                mesh=self.mesh,
                axis_name="model",
                out_dtype=fuse.dtype,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                k_local=k_local,
                pipeline_mode=envs.FUSED_AR_PIPELINE_MODE or "5stage",
            )
            if fuse.ndim > 2:
                out = out.reshape((*fuse.shape[:-1], self.hidden_size))
            return out
        return self.down_proj(fuse)


class Qwen3DecoderLayer(Qwen2DecoderLayer):

    def __init__(self,
                 config: Qwen3Config,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 quant_config: VllmQuantConfig,
                 additional_config: dict | None = None,
                 prefix: str = ""):
        rms_norm_eps = config.rms_norm_eps
        hidden_size = config.hidden_size

        self.input_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            dtype=dtype,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".input_layernorm",
        )
        self.self_attn = Qwen3Attention(config=config,
                                        dtype=dtype,
                                        rng=rng,
                                        mesh=mesh,
                                        kv_cache_dtype=kv_cache_dtype,
                                        quant_config=quant_config,
                                        additional_config=additional_config,
                                        prefix=prefix + ".self_attn")
        self.post_attention_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            dtype=dtype,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".post_attention_layernorm",
        )
        self.mlp = Qwen3MLP(
            config=config,
            dtype=dtype,
            rng=rng,
            mesh=mesh,
            quant_config=quant_config,
            additional_config=additional_config,
            prefix=prefix + ".mlp",
        )
        self.mesh = mesh

    def __call__(
        self,
        kv_cache: jax.Array,
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        if envs.USE_FUSED_RMSNORM_FP8:
            x_2d = x.reshape((-1, x.shape[-1])) if x.ndim > 2 else x
            gamma_pre = self.input_layernorm.weight[...]
            eps_pre = getattr(self.input_layernorm, "epsilon", 1e-6)

            # 1. Pre-Attention Fused RMSNorm + FP8 Cast
            pre_res = _apply_fused_rmsnorm_fp8(
                x_2d,
                gamma_pre,
                mesh=self.mesh,
                residual=None,
                return_residual=False,
                eps=eps_pre,
                block_k=envs.FUSED_RMSNORM_BLOCK_K,
            )
            if pre_res is not None:
                x_q, scale = pre_res
                if QArray is not None:
                    hidden_states = QArray(qvalue=x_q, scale=scale, qtype=jnp.float8_e4m3fn)
                    if x.ndim > 2:
                        hidden_states = hidden_states.reshape(x.shape)
                else:
                    hidden_states = (x_q.astype(jnp.bfloat16) * scale)
                    if x.ndim > 2:
                        hidden_states = hidden_states.reshape(x.shape)
            else:
                hidden_states = self.input_layernorm(x)

            # 2. Self-Attention Block
            kv_cache, attn_output = self.self_attn(
                kv_cache,
                hidden_states,
                attention_metadata,
            )

            # 3. Post-Attention Fused RMSNorm + Residual Addition + FP8 Cast
            attn_2d = attn_output.reshape((-1, attn_output.shape[-1])) if attn_output.ndim > 2 else attn_output
            gamma_post = self.post_attention_layernorm.weight[...]
            eps_post = getattr(self.post_attention_layernorm, "epsilon", 1e-6)
            post_res = _apply_fused_rmsnorm_fp8(
                attn_2d,
                gamma_post,
                mesh=self.mesh,
                residual=x_2d,
                return_residual=True,
                eps=eps_post,
                block_k=envs.FUSED_RMSNORM_BLOCK_K,
            )
            if post_res is not None:
                post_q, post_scale, updated_residual = post_res
                if QArray is not None:
                    mlp_in = QArray(qvalue=post_q, scale=post_scale, qtype=jnp.float8_e4m3fn)
                    if x.ndim > 2:
                        mlp_in = mlp_in.reshape(x.shape)
                    if x.ndim > 2:
                        updated_residual = updated_residual.reshape(x.shape)
                else:
                    mlp_in = (post_q.astype(jnp.bfloat16) * post_scale)
                    if x.ndim > 2:
                        mlp_in = mlp_in.reshape(x.shape)
                        updated_residual = updated_residual.reshape(x.shape)
            else:
                attn_output += x
                updated_residual = attn_output
                mlp_in = self.post_attention_layernorm(attn_output)

            # 4. MLP Block
            outputs = self.mlp(mlp_in)
            outputs = updated_residual + outputs
            return kv_cache, outputs

        hidden_states = self.input_layernorm(x)
        kv_cache, attn_output = self.self_attn(
            kv_cache,
            hidden_states,
            attention_metadata,
        )
        attn_output += x

        residual = attn_output
        attn_output = self.post_attention_layernorm(attn_output)
        outputs = self.mlp(attn_output)
        outputs = residual + outputs
        return kv_cache, outputs


class Qwen3Model(Qwen2Model):

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 prefix: str = "model") -> None:
        self.mesh = mesh
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        vocab_size = model_config.get_vocab_size()
        dtype = model_config.dtype
        rms_norm_eps = hf_config.rms_norm_eps
        hidden_size = hf_config.hidden_size

        self.is_first_rank = get_pp_group().is_first_rank
        self.is_last_rank = get_pp_group().is_last_rank

        tp_size = vllm_config.parallel_config.tensor_parallel_size if vllm_config.parallel_config is not None else 1
        padded_vocab_size = utils.align_to(vocab_size, tp_size)

        if self.is_first_rank or (hf_config.tie_word_embeddings
                                  and self.is_last_rank):
            self.embed_tokens = JaxEmbed(
                num_embeddings=padded_vocab_size,
                features=hidden_size,
                dtype=dtype,
                param_dtype=dtype,
                embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
                rngs=rng,
                quant_config=vllm_config.quant_config,
                prefix=prefix + ".embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            hf_config.num_hidden_layers,
            lambda layer_index: Qwen3DecoderLayer(
                config=hf_config,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
                # TODO (jacobplatin): we should refactor this to pass a dtype (or config) directly
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                quant_config=vllm_config.quant_config,
                additional_config=getattr(vllm_config, "additional_config", None),
                prefix=f"{prefix}.layers.{layer_index}",
            ))
        if self.is_last_rank:
            self.norm = JaxRmsNorm(
                hidden_size,
                epsilon=rms_norm_eps,
                dtype=dtype,
                param_dtype=dtype,
                scale_init=nnx.with_partitioning(init_fn, (None, )),
                rngs=rng,
                quant_config=vllm_config.quant_config,
                prefix=prefix + ".norm",
            )
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers = []
        spec_config = getattr(vllm_config, "speculative_config", None)
        if spec_config and spec_config.method == "dflash":
            self.aux_hidden_state_layers = self.get_dflash_aux_hidden_state_layers(
                vllm_config)

    def get_dflash_aux_hidden_state_layers(self, vllm_config):
        spec_config = getattr(vllm_config, "speculative_config", None)
        if spec_config is None or spec_config.draft_model_config is None:
            return []
        draft_hf_config = spec_config.draft_model_config.hf_config
        dflash_config = getattr(draft_hf_config, "dflash_config", {})
        target_layer_ids = dflash_config.get("target_layer_ids", None)
        if target_layer_ids is not None:
            return [i for i in target_layer_ids]
        hf_config = vllm_config.model_config.hf_config
        num_target_layers = getattr(draft_hf_config, "num_target_layers",
                                    hf_config.num_hidden_layers)
        num_layers = hf_config.num_hidden_layers
        return list(range(num_layers - num_target_layers, num_layers))

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: Optional[jax.Array],
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array]]:
        from itertools import islice
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embed_tokens(input_ids)

        aux_hidden_states = []
        for i, layer in enumerate(
                islice(self.layers, self.start_layer, self.end_layer)):
            kv_cache = kv_caches[i]
            kv_cache, x = layer(
                kv_cache,
                x,
                attention_metadata,
            )
            kv_caches[i] = kv_cache
            if i in self.aux_hidden_state_layers:
                aux_hidden_states.append(x)
        if envs.USE_FUSED_RMSNORM_FP8 and isinstance(self.norm, JaxRmsNorm):
            x_2d = x.reshape((-1, x.shape[-1])) if x.ndim > 2 else x
            gamma = self.norm.weight[...]
            eps = getattr(self.norm, "epsilon", 1e-6)
            norm_res = _apply_fused_rmsnorm_fp8(
                x_2d,
                gamma,
                mesh=self.mesh,
                residual=None,
                return_residual=False,
                eps=eps,
                block_k=envs.FUSED_RMSNORM_BLOCK_K,
            )
            if norm_res is not None:
                x_q, scale = norm_res
                x_norm = (x_q.astype(jnp.bfloat16) * scale)
                x = x_norm.reshape(x.shape) if x.ndim > 2 else x_norm
            else:
                x = self.norm(x)
        else:
            x = self.norm(x)
        return kv_caches, x, aux_hidden_states


class Qwen3ForCausalLM(JaxModule, LoadableWithIterator):
    packed_modules_mapping = {
        "kv_proj": [
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.model = Qwen3Model(
            vllm_config=vllm_config,
            rng=rng,
            mesh=mesh,
            prefix="model",
        )
        model_config = vllm_config.model_config
        is_pooling = vllm_config.model_config.runner_type == "pooling"
        if not model_config.hf_config.tie_word_embeddings and not is_pooling:
            if self.model.is_last_rank:
                vocab_size = model_config.get_vocab_size()
                tp_size = vllm_config.parallel_config.tensor_parallel_size if vllm_config.parallel_config is not None else 1
                padded_vocab_size = utils.align_to(vocab_size, tp_size)
                hidden_size = model_config.hf_config.hidden_size
                self.lm_head = JaxLmHead(
                    hidden_size=hidden_size,
                    vocab_size=padded_vocab_size,
                    dtype=model_config.dtype,
                    param_dtype=model_config.dtype,
                    rngs=rng,
                    prefix="lm_head",
                )
            else:
                self.lm_head = PPMissingLayer()
        else:
            self.lm_head = PPMissingLayer()

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        _input_positions=None,
        _layer_name_to_kv_cache=None,
        _lora_metadata=None,
        intermediate_tensors: JaxIntermediateTensors | None = None,
        is_first_rank: bool = True,
        is_last_rank: bool = True,
        *args,
    ) -> Tuple[List[jax.Array], jax.Array | JaxIntermediateTensors,
               List[jax.Array], Optional[jax.Array]]:
        if not is_first_rank:
            assert intermediate_tensors is not None
            inputs_embeds = intermediate_tensors["hidden_states"]
        kv_caches, x, aux_hidden_states = self.model(
            kv_caches,
            input_ids,
            attention_metadata,
            inputs_embeds,
        )
        if not is_last_rank:
            x = JaxIntermediateTensors(tensors={"hidden_states": x}, )
        return kv_caches, x, aux_hidden_states, None

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        # Only use lm_head if it's a real projection layer (not a PPMissingLayer placeholder)
        if hasattr(self,
                   'lm_head') and not isinstance(self.lm_head, PPMissingLayer):
            return self.lm_head(hidden_states)

        assert isinstance(self.model.embed_tokens, JaxEmbed)
        return self.model.embed_tokens.decode(hidden_states)
