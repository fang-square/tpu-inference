# Copyright 2026 Google LLC
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

"""Tests for Fused SwiGLU + FP8 Dynamic Quantization Pallas TPU Kernel."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.kernels.swiglu.fused_swiglu_pallas import (
    fused_swiglu_pallas,
)


def reference_swiglu(
    lhs_q: jax.Array,
    lhs_scale: jax.Array,
    w_gate_q: jax.Array,
    w_gate_scale: jax.Array,
    w_up_q: jax.Array,
    w_up_scale: jax.Array,
    quant_out: bool = True,
    quant_mode: str = "separate_channelwise",
    subchannel_k: int = 512,
    quant_max: float = 448.0,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Reference un-fused SwiGLU computation with optional FP8 quantization."""
  acc_gate = jnp.dot(
      lhs_q.astype(jnp.float32),
      w_gate_q.astype(jnp.float32),
      preferred_element_type=jnp.float32,
  )
  acc_up = jnp.dot(
      lhs_q.astype(jnp.float32),
      w_up_q.astype(jnp.float32),
      preferred_element_type=jnp.float32,
  )

  gate_scaled = acc_gate.astype(jnp.bfloat16) * (lhs_scale * w_gate_scale)
  up_scaled = acc_up.astype(jnp.bfloat16) * (lhs_scale * w_up_scale)
  res = jax.nn.silu(gate_scaled) * up_scaled

  if not quant_out:
    return res.astype(jnp.bfloat16)

  if quant_mode in ("separate_channelwise", "channelwise_separate"):
    abs_max = jnp.max(jnp.abs(res), axis=0, keepdims=True)  # [1, N]
    scale = jnp.maximum(abs_max / quant_max, 1e-12)
    out_q = jnp.clip(res / scale, -quant_max, quant_max).astype(
        jnp.float8_e4m3fn
    )
    out_scale = scale.astype(jnp.bfloat16)
    return out_q, out_scale
  elif quant_mode == "channelwise":
    abs_max = jnp.max(jnp.abs(res), axis=0, keepdims=True)
    scale = jnp.maximum(abs_max / quant_max, 1e-12)
    out_q = jnp.clip(res / scale, -quant_max, quant_max).astype(
        jnp.float8_e4m3fn
    )
    out_scale = scale.astype(jnp.bfloat16)
    return out_q, out_scale
  else:
    m, n = res.shape
    if subchannel_k <= 0 or subchannel_k >= n:
      abs_max = jnp.max(jnp.abs(res), axis=-1, keepdims=True)
      scale = jnp.maximum(abs_max / quant_max, 1e-12)
      out_q = jnp.clip(res / scale, -quant_max, quant_max).astype(
          jnp.float8_e4m3fn
      )
      out_scale = scale.astype(jnp.bfloat16)
      return out_q, out_scale
    else:
      num_blocks = n // subchannel_k
      res_blocked = jnp.reshape(res, (m, num_blocks, subchannel_k))
      abs_max = jnp.max(jnp.abs(res_blocked), axis=-1, keepdims=True)
      scale = jnp.maximum(abs_max / quant_max, 1e-12)
      res_scaled = res_blocked / scale
      out_q = jnp.reshape(
          jnp.clip(res_scaled, -quant_max, quant_max), (m, n)
      ).astype(jnp.float8_e4m3fn)
      out_scale = jnp.reshape(scale, (m, num_blocks)).astype(jnp.bfloat16)
      return out_q, out_scale


class TestFusedSwiGLUKernel:

  @pytest.mark.parametrize(
      "m,k,n,tile_m,tile_n,tile_k",
      [
          (512, 5120, 1024, 256, 512, 5120),
          (1024, 5120, 2048, 512, 512, 5120),
          (2048, 5120, 4096, 1024, 512, 5120),
      ],
  )
  def test_grid_mode_separate_channelwise_correctness(
      self, m, k, n, tile_m, tile_n, tile_k
  ):
    key = jax.random.key(42)
    k_lhs, k_ls, k_wg, k_wgs, k_wu, k_wus = jax.random.split(key, 6)

    # Random FP8 inputs
    lhs_f32 = jax.random.normal(k_lhs, (m, k), dtype=jnp.float32)
    lhs_q = jnp.clip(lhs_f32, -448.0, 448.0).astype(jnp.float8_e4m3fn)
    lhs_scale = jax.random.uniform(k_ls, (m, 1), dtype=jnp.bfloat16) * 0.01 + 0.001

    w_gate_f32 = jax.random.normal(k_wg, (k, n), dtype=jnp.float32)
    w_gate_q = jnp.clip(w_gate_f32, -448.0, 448.0).astype(jnp.float8_e4m3fn)
    w_gate_scale = jax.random.uniform(k_wgs, (1, n), dtype=jnp.bfloat16) * 0.01 + 0.001

    w_up_f32 = jax.random.normal(k_wu, (k, n), dtype=jnp.float32)
    w_up_q = jnp.clip(w_up_f32, -448.0, 448.0).astype(jnp.float8_e4m3fn)
    w_up_scale = jax.random.uniform(k_wus, (1, n), dtype=jnp.bfloat16) * 0.01 + 0.001

    # 1. Reference
    ref_q, ref_s = reference_swiglu(
        lhs_q,
        lhs_scale,
        w_gate_q,
        w_gate_scale,
        w_up_q,
        w_up_scale,
        quant_out=True,
        quant_mode="separate_channelwise",
    )

    # 2. Fused Pallas Kernel (Grid Mode + Separate Channelwise)
    pallas_fn = jax.jit(
        lambda _lq, _ls, _wgq, _wgs, _wuq, _wus: fused_swiglu_pallas(
            _lq,
            _ls,
            _wgq,
            _wgs,
            _wuq,
            _wus,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            quant_out=True,
            quant_mode="separate_channelwise",
            pipeline_mode="grid",
        )
    )
    pal_q, pal_s = pallas_fn(
        lhs_q, lhs_scale, w_gate_q, w_gate_scale, w_up_q, w_up_scale
    )

    # Dequantize both
    ref_dequant = ref_q.astype(jnp.float32) * ref_s.astype(jnp.float32)
    pal_dequant = pal_q.astype(jnp.float32) * pal_s.astype(jnp.float32)

    ref_np = np.array(ref_dequant, dtype=np.float32)
    pal_np = np.array(pal_dequant, dtype=np.float32)

    cos_sim = np.dot(ref_np.flatten(), pal_np.flatten()) / (
        np.linalg.norm(ref_np.flatten()) * np.linalg.norm(pal_np.flatten())
        + 1e-12
    )
    assert (
        cos_sim > 0.999
    ), f"Cosine similarity {cos_sim} below threshold 0.999"

    scale_diff = np.max(np.abs(np.array(ref_s, dtype=np.float32) - np.array(pal_s, dtype=np.float32)))
    assert scale_diff < 1e-4, f"Scale max diff {scale_diff} too large"

  def test_grid_mode_no_quant(self):
    m, k, n = 512, 5120, 1024
    key = jax.random.key(123)
    k_lhs, k_ls, k_wg, k_wgs, k_wu, k_wus = jax.random.split(key, 6)

    lhs_q = jnp.clip(jax.random.normal(k_lhs, (m, k), dtype=jnp.float32), -448.0, 448.0).astype(jnp.float8_e4m3fn)
    lhs_scale = jax.random.uniform(k_ls, (m, 1), dtype=jnp.bfloat16) * 0.01 + 0.001
    w_gate_q = jnp.clip(jax.random.normal(k_wg, (k, n), dtype=jnp.float32), -448.0, 448.0).astype(jnp.float8_e4m3fn)
    w_gate_scale = jax.random.uniform(k_wgs, (1, n), dtype=jnp.bfloat16) * 0.01 + 0.001
    w_up_q = jnp.clip(jax.random.normal(k_wu, (k, n), dtype=jnp.float32), -448.0, 448.0).astype(jnp.float8_e4m3fn)
    w_up_scale = jax.random.uniform(k_wus, (1, n), dtype=jnp.bfloat16) * 0.01 + 0.001

    ref_out = reference_swiglu(
        lhs_q,
        lhs_scale,
        w_gate_q,
        w_gate_scale,
        w_up_q,
        w_up_scale,
        quant_out=False,
    )

    pal_out = jax.jit(
        lambda _lq, _ls, _wgq, _wgs, _wuq, _wus: fused_swiglu_pallas(
            _lq,
            _ls,
            _wgq,
            _wgs,
            _wuq,
            _wus,
            tile_m=256,
            tile_n=512,
            tile_k=5120,
            quant_out=False,
            pipeline_mode="grid",
        )
    )(lhs_q, lhs_scale, w_gate_q, w_gate_scale, w_up_q, w_up_scale)

    ref_np = np.array(ref_out, dtype=np.float32)
    pal_np = np.array(pal_out, dtype=np.float32)

    cos_sim = np.dot(ref_np.flatten(), pal_np.flatten()) / (
        np.linalg.norm(ref_np.flatten()) * np.linalg.norm(pal_np.flatten())
        + 1e-12
    )
    assert cos_sim > 0.99999, f"Cosine similarity {cos_sim} below 0.99999"
