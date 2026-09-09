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
      raise ValueError(
          f"Row dimension M ({m}) must be divisible by block_m ({block_m})"
      )

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
