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

"""Fused RMSNorm + FP8 Dynamic Quantization Cast Pallas TPU Kernel.

This kernel fuses RMSNorm and FP8 dynamic per-token / sub-channel quantization
into a single streaming HBM pass on TPU:
  1. Loads input activation tile X [B_M, K] (+ optional residual) into VMEM.
  2. Computes FP32 variance and RMSNorm: x_norm = (x * rsqrt(mean(x^2) + eps)) * gamma.
  3. Computes dynamic scale factor s = max(|x_norm|) / 448.0 per token (or sub-channel).
  4. Saturates, clamps, and casts directly to float8_e4m3fn in VMEM registers.
  5. Streams quantized activations X_q [B_M, K] and scale s [B_M, num_scales] to HBM.
"""

import functools
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def _fused_rmsnorm_quant_kernel(
    *args,
    eps: float = 1e-6,
    quant_max: float = 448.0,
    has_residual: bool = False,
    return_residual: bool = False,
    block_k: int | None = None,
):
  """Inner Pallas kernel executing on TPU VPU vector units."""
  arg_idx = 0
  x_ref = args[arg_idx]
  arg_idx += 1
  gamma_ref = args[arg_idx]
  arg_idx += 1
  if has_residual:
    residual_ref = args[arg_idx]
    arg_idx += 1
  else:
    residual_ref = None

  out_q_ref = args[arg_idx]
  arg_idx += 1
  out_scale_ref = args[arg_idx]
  arg_idx += 1
  if return_residual:
    out_res_ref = args[arg_idx]
    arg_idx += 1
  else:
    out_res_ref = None

  x = x_ref[...]
  if has_residual and residual_ref is not None:
    x = x + residual_ref[...]
    if return_residual and out_res_ref is not None:
      out_res_ref[...] = x.astype(jnp.bfloat16)

  # 1. High-precision FP32 variance and RMS normalization
  x_f32 = x.astype(jnp.float32)
  var = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
  rsqrt_var = jax.lax.rsqrt(var + eps)
  gamma_f32 = gamma_ref[...].astype(jnp.float32)
  x_norm = (x_f32 * rsqrt_var) * gamma_f32

  # 2. Dynamic scale computation and quantization
  if block_k is None or block_k >= x_norm.shape[-1]:
    # Standard per-token / row-wise dynamic quantization: [B_M, 1]
    x_abs_max = jnp.max(jnp.abs(x_norm), axis=-1, keepdims=True)
    scale = jnp.maximum(x_abs_max / quant_max, 1e-12)
    x_scaled = x_norm / scale
    x_clamped = jnp.clip(x_scaled, -quant_max, quant_max)
    out_scale_ref[...] = scale.astype(jnp.bfloat16)
  else:
    # Sub-channel block quantization: [B_M, K // block_k]
    bm, k = x_norm.shape
    num_blocks = k // block_k
    x_norm_blocked = jnp.reshape(x_norm, (bm, num_blocks, block_k))
    x_abs_max = jnp.max(jnp.abs(x_norm_blocked), axis=-1, keepdims=True)
    scale = jnp.maximum(x_abs_max / quant_max, 1e-12)
    x_scaled = x_norm_blocked / scale
    x_clamped = jnp.clip(x_scaled, -quant_max, quant_max)
    x_clamped = jnp.reshape(x_clamped, (bm, k))
    out_scale_ref[...] = jnp.reshape(scale, (bm, num_blocks)).astype(
        jnp.bfloat16
    )

  # 3. Direct cast to float8_e4m3fn in VMEM
  out_q_ref[...] = x_clamped.astype(jnp.float8_e4m3fn)


def fused_rmsnorm_fp8_quant(
    x: jax.Array,
    gamma: jax.Array,
    residual: jax.Array | None = None,
    *,
    block_m: int = 512,
    block_k: int | None = None,
    eps: float = 1e-6,
    quant_max: float = 448.0,
    return_residual: bool = False,
    vmem_limit_bytes: int = 128 * 1024 * 1024,
) -> tuple[jax.Array, ...] | jax.Array:
  """Fused RMSNorm and FP8 Dynamic Quantization Pallas TPU kernel.

  Args:
    x: Input activation tensor of shape [M, K] in bfloat16.
    gamma: RMSNorm scale weights of shape [K] in bfloat16 or float32.
    residual: Optional residual tensor of shape [M, K] in bfloat16.
    block_m: Token tiling block size along sequence dimension M (default: 512).
    block_k: Optional sub-channel quantization block size along K (default: None
      for per-token row-wise quantization).
    eps: Small epsilon for numerical stability in RMSNorm (default: 1e-6).
    quant_max: Maximum representable magnitude for FP8 dtype (default: 448.0 for
      e4m3fn).
    return_residual: If True and residual is given, returns the updated residual
      (x + residual) along with (out_q, out_scale).
    vmem_limit_bytes: TPU VMEM memory allocation limit.

  Returns:
    tuple of (out_q, out_scale) where:
      out_q: Quantized activations [M, K] in float8_e4m3fn.
      out_scale: Scale factor tensor [M, num_scales] in bfloat16.
    If return_residual is True, returns (out_q, out_scale, out_residual).
  """
  m, k = x.shape
  if m % block_m != 0:
    raise ValueError(f"Sequence length M ({m}) must be divisible by block_m ({block_m})")

  grid_m = m // block_m
  has_residual = residual is not None
  has_res_out = has_residual and return_residual

  num_scales = 1 if (block_k is None or block_k >= k) else (k // block_k)

  in_specs = [
      pl.BlockSpec((block_m, k), lambda i: (i, 0)),  # x
      pl.BlockSpec((k,), lambda i: (0,)),  # gamma
  ]
  if has_residual:
    in_specs.append(pl.BlockSpec((block_m, k), lambda i: (i, 0)))  # residual

  out_specs = [
      pl.BlockSpec((block_m, k), lambda i: (i, 0)),  # out_q
      pl.BlockSpec((block_m, num_scales), lambda i: (i, 0)),  # out_scale
  ]
  out_shapes = [
      jax.ShapeDtypeStruct((m, k), jnp.float8_e4m3fn),
      jax.ShapeDtypeStruct((m, num_scales), jnp.bfloat16),
  ]

  if has_res_out:
    out_specs.append(pl.BlockSpec((block_m, k), lambda i: (i, 0)))
    out_shapes.append(jax.ShapeDtypeStruct((m, k), jnp.bfloat16))

  kernel_fn = functools.partial(
      _fused_rmsnorm_quant_kernel,
      eps=eps,
      quant_max=quant_max,
      has_residual=has_residual,
      return_residual=has_res_out,
      block_k=block_k,
  )

  args = [x, gamma]
  if has_residual:
    args.append(residual)

  pallas_out = pl.pallas_call(
      kernel_fn,
      grid=(grid_m,),
      in_specs=in_specs,
      out_specs=out_specs,
      out_shape=out_shapes,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel",),
          vmem_limit_bytes=vmem_limit_bytes,
          disable_bounds_checks=True,
      ),
  )(*args)

  return pallas_out
