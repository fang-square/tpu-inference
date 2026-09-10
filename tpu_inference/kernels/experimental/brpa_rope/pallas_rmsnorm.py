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

"""Dedicated 2D Pallas RMSNorm TPU Kernel.

Streams [N, D] activations (e.g. N=131072, D=128) through TPU VMEM in row-major
order,
bypassing XLA's automatic sublane tiling heuristics and eliminating intermediate
HBM transpose copies. Normalizes in FP32 vector registers and writes out
directly
in the layout expected by RPAm.
"""

import functools
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def _rmsnorm_2d_kernel(
    x_ref,
    *args,
    eps: float = 1e-6,
    has_gamma: bool = False,
):
  """Inner Pallas 2D RMSNorm kernel executing on TPU VPU vector units."""
  if has_gamma:
    gamma_ref = args[0]
    out_ref = args[1]
  else:
    gamma_ref = None
    out_ref = args[0]

  x = x_ref[...]
  x_f32 = x.astype(jnp.float32)
  var = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
  rsqrt_var = jax.lax.rsqrt(var + eps)

  if has_gamma and gamma_ref is not None:
    gamma_f32 = gamma_ref[...].astype(jnp.float32)
    normed = (x_f32 * rsqrt_var) * gamma_f32
  else:
    normed = x_f32 * rsqrt_var

  out_ref[...] = normed.astype(x.dtype)


def pallas_2d_rmsnorm(
    x: jax.Array,
    gamma: jax.Array | None = None,
    *,
    eps: float = 1e-6,
    block_m: int = 1024,
    vmem_limit_bytes: int = 16 * 1024 * 1024,
) -> jax.Array:
  """Dedicated 2D Pallas RMSNorm Kernel on TPU.

  Args:
    x: Input tensor of shape [..., D] (e.g., [131072, 128] or [4, 4096, 8,
      128]).
    gamma: Optional scale parameter of shape [D]. If None, unit scaling is
      applied.
    eps: Epsilon for numerical stability.
    block_m: Tile size along the row dimension M (default: 1024).
    vmem_limit_bytes: Maximum VMEM allocation for the kernel.

  Returns:
    Normalized tensor of identical shape and dtype as x.
  """
  has_tpu = any("tpu" in d.device_kind.lower() for d in jax.devices())
  if not has_tpu:
    x_f32 = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    rsqrt_var = jax.lax.rsqrt(var + eps)
    if gamma is not None:
      normed = (x_f32 * rsqrt_var) * gamma.astype(jnp.float32)
    else:
      normed = x_f32 * rsqrt_var
    return normed.astype(x.dtype)

  orig_shape = x.shape
  d = orig_shape[-1]
  x_2d = x.reshape(-1, d)
  m, k = x_2d.shape

  if m % block_m != 0:
    for candidate in [1024, 512, 256, 128, 64, 32, 16]:
      if m % candidate == 0 and candidate <= m:
        block_m = candidate
        break
    else:
      x_f32 = x.astype(jnp.float32)
      var = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
      rsqrt_var = jax.lax.rsqrt(var + eps)
      if gamma is not None:
        normed = (x_f32 * rsqrt_var) * gamma.astype(jnp.float32)
      else:
        normed = x_f32 * rsqrt_var
      return normed.astype(x.dtype)

  grid_m = m // block_m
  has_gamma = gamma is not None

  in_specs = [
      pl.BlockSpec((block_m, k), lambda i: (i, 0)),
  ]
  args = [x_2d]

  if has_gamma:
    in_specs.append(pl.BlockSpec((k,), lambda i: (0,)))
    args.append(gamma.reshape(k))

  out_specs = pl.BlockSpec((block_m, k), lambda i: (i, 0))

  kernel_fn = functools.partial(
      _rmsnorm_2d_kernel,
      eps=eps,
      has_gamma=has_gamma,
  )

  out_2d = pl.pallas_call(
      kernel_fn,
      grid=(grid_m,),
      in_specs=in_specs,
      out_specs=out_specs,
      out_shape=jax.ShapeDtypeStruct((m, k), x.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel",),
          vmem_limit_bytes=vmem_limit_bytes,
          disable_bounds_checks=True,
      ),
  )(*args)

  return out_2d.reshape(orig_shape)


def compute_rope_cos_sin(
    total_tokens: int,
    head_dim: int,
    theta: float = 1000000.0,
    positions: jax.Array | None = None,
) -> jax.Array:
  """Computes [total_tokens, head_dim] concatenated [cos, sin] table for RoPE."""
  half_d = head_dim // 2
  dim_indices = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
  inv_freq = 1.0 / (theta ** (dim_indices / float(head_dim)))
  if positions is None:
    positions = jnp.arange(total_tokens, dtype=jnp.float32)
  else:
    positions = positions.astype(jnp.float32)
  freqs = jnp.outer(positions, inv_freq)  # [total_tokens, half_d]
  cos = jnp.cos(freqs)
  sin = jnp.sin(freqs)
  return jnp.concatenate([cos, sin], axis=-1).astype(jnp.float32)


def _rmsnorm_rope_kernel(
    x_ref,
    *args,
    eps: float = 1e-6,
    ordering: str = "split",
    has_gamma: bool = True,
):
  """Inner Pallas kernel fusing 2D RMSNorm and RoPE vector rotation in VMEM."""
  arg_idx = 0
  if has_gamma:
    gamma_ref = args[arg_idx]
    arg_idx += 1
  else:
    gamma_ref = None

  cos_sin_ref = args[arg_idx]
  arg_idx += 1
  out_ref = args[arg_idx]

  x = x_ref[...]  # shape: [1, block_t, g, d]
  x_f32 = x.astype(jnp.float32)

  # 1. High-precision FP32 RMSNorm along head_dim (axis -1)
  var = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
  rsqrt_var = jax.lax.rsqrt(var + eps)
  if has_gamma and gamma_ref is not None:
    gamma_f32 = gamma_ref[...].astype(jnp.float32)
    normed = (x_f32 * rsqrt_var) * gamma_f32
  else:
    normed = x_f32 * rsqrt_var

  # 2. In-Register RoPE rotation in FP32 vector registers
  cos_sin = cos_sin_ref[...]  # [block_t, d]
  d = x.shape[-1]
  half_d = d // 2
  cos = cos_sin[:, :half_d]  # [block_t, half_d]
  sin = cos_sin[:, half_d:]  # [block_t, half_d]

  # Broadcast across [1, block_t, 1, half_d] to match [1, block_t, g, d]
  cos_b = jnp.expand_dims(cos, (0, 2))
  sin_b = jnp.expand_dims(sin, (0, 2))

  if ordering == "split":
    cos_full = jnp.concatenate([cos_b, cos_b], axis=-1)
    sin_full = jnp.concatenate([-sin_b, sin_b], axis=-1)
    x_swap = jnp.concatenate(
        [normed[..., half_d:], normed[..., :half_d]], axis=-1
    )
    x_rot = normed * cos_full + x_swap * sin_full
  else:
    cos_full = jnp.repeat(cos_b, 2, axis=-1)
    sin_full = jnp.stack([-sin_b, sin_b], axis=-1).reshape(1, x.shape[1], 1, d)
    x_swap = jnp.stack([normed[..., 1::2], normed[..., 0::2]], axis=-1).reshape(
        normed.shape
    )
    x_rot = normed * cos_full + x_swap * sin_full

  # 3. Direct cast to original dtype (BF16)
  out_ref[...] = x_rot.astype(x.dtype)


def pallas_2d_rmsnorm_rope(
    x: jax.Array,
    gamma: jax.Array | None = None,
    positions: jax.Array | None = None,
    *,
    theta: float = 1000000.0,
    ordering: str = "split",
    eps: float = 1e-6,
    block_t: int = 64,
    vmem_limit_bytes: int = 16 * 1024 * 1024,
) -> jax.Array:
  """Dedicated Fused 2D Pallas RMSNorm + RoPE Kernel on TPU.

  Normalizes activations along the head dimension D=128 and immediately applies
  Rotary Position Embedding (RoPE) in FP32 VPU vector registers within a single
  streaming pass, eliminating intermediate HBM round-trips and bypassing RoPE
  computation inside downstream attention kernels.

  Args:
    x: Input query activation tensor. Supported shapes: - 4D Head-Major: [H_kv,
      T, G, D] (e.g. [4, 4096, 8, 128]) - 3D Token-Major: [T, H_q, D] (e.g.
      [4096, 32, 128]) - 2D: [T, D]
    gamma: Optional RMSNorm scale weights of shape [D].
    positions: Optional token position indices of shape [T]. If None, defaults
      to [0, 1, ..., T-1].
    theta: RoPE base frequency (default: 1000000.0).
    ordering: RoPE rotation ordering ("split" or "interleaved", default
      "split").
    eps: Epsilon for numerical stability (default: 1e-6).
    block_t: Tiling block size along token dimension T (default: 64).
    vmem_limit_bytes: Maximum VMEM allocation limit.

  Returns:
    Normalized and rotary-embedded tensor with identical shape and dtype as x.
  """
  orig_shape = x.shape
  d = orig_shape[-1]
  half_d = d // 2

  if x.ndim == 4:
    m_outer, t, g, _ = orig_shape
    x_4d = x
  elif x.ndim == 3:
    t, g, _ = orig_shape
    m_outer = 1
    x_4d = x.reshape(1, t, g, d)
  elif x.ndim == 2:
    t, _ = orig_shape
    m_outer = 1
    g = 1
    x_4d = x.reshape(1, t, 1, d)
  else:
    raise ValueError(f"Unsupported input rank {x.ndim}: shape={orig_shape}")

  # Adjust block_t if t is smaller than default block_t
  actual_block_t = min(block_t, t)
  if t % actual_block_t != 0:
    # Fallback to GCD divisor
    for candidate in [32, 16, 8, 4, 2, 1]:
      if t % candidate == 0 and candidate <= block_t:
        actual_block_t = candidate
        break

  cos_sin = compute_rope_cos_sin(t, d, theta=theta, positions=positions)

  has_tpu = any("tpu" in d_dev.device_kind.lower() for d_dev in jax.devices())
  if not has_tpu:
    x_f32 = x_4d.astype(jnp.float32)
    var = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    rsqrt_var = jax.lax.rsqrt(var + eps)
    if gamma is not None:
      normed = (x_f32 * rsqrt_var) * gamma.astype(jnp.float32)
    else:
      normed = x_f32 * rsqrt_var

    cos = cos_sin[:, :half_d]
    sin = cos_sin[:, half_d:]
    cos_b = jnp.expand_dims(cos, (0, 2))
    sin_b = jnp.expand_dims(sin, (0, 2))

    if ordering == "split":
      cos_full = jnp.concatenate([cos_b, cos_b], axis=-1)
      sin_full = jnp.concatenate([-sin_b, sin_b], axis=-1)
      x_swap = jnp.concatenate(
          [normed[..., half_d:], normed[..., :half_d]], axis=-1
      )
      x_rot = normed * cos_full + x_swap * sin_full
    else:
      cos_full = jnp.repeat(cos_b, 2, axis=-1)
      sin_full = jnp.stack([-sin_b, sin_b], axis=-1).reshape(1, t, 1, d)
      x_swap = jnp.stack(
          [normed[..., 1::2], normed[..., 0::2]], axis=-1
      ).reshape(normed.shape)
      x_rot = normed * cos_full + x_swap * sin_full

    return x_rot.reshape(orig_shape).astype(x.dtype)

  grid = (m_outer, t // actual_block_t)
  has_gamma = gamma is not None

  in_specs = [
      pl.BlockSpec(
          (1, actual_block_t, g, d), lambda m_idx, t_idx: (m_idx, t_idx, 0, 0)
      ),
  ]
  args = [x_4d]

  if has_gamma:
    in_specs.append(pl.BlockSpec((d,), lambda m_idx, t_idx: (0,)))
    args.append(gamma.reshape(d))

  in_specs.append(
      pl.BlockSpec((actual_block_t, d), lambda m_idx, t_idx: (t_idx, 0))
  )
  args.append(cos_sin)

  out_specs = pl.BlockSpec(
      (1, actual_block_t, g, d), lambda m_idx, t_idx: (m_idx, t_idx, 0, 0)
  )

  kernel_fn = functools.partial(
      _rmsnorm_rope_kernel,
      eps=eps,
      ordering=ordering,
      has_gamma=has_gamma,
  )

  out_4d = pl.pallas_call(
      kernel_fn,
      grid=grid,
      in_specs=in_specs,
      out_specs=out_specs,
      out_shape=jax.ShapeDtypeStruct(x_4d.shape, x.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel"),
          vmem_limit_bytes=vmem_limit_bytes,
          disable_bounds_checks=True,
      ),
  )(*args)

  return out_4d.reshape(orig_shape)
