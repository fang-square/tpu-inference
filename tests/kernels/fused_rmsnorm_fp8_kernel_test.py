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

"""Tests for Fused RMSNorm + FP8 Dynamic Quantization Pallas TPU Kernel."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.kernels.fused_rmsnorm_quant.fused_rmsnorm_fp8 import (
    fused_rmsnorm_fp8_quant,
)


def reference_rmsnorm_fp8(
    x: jax.Array,
    gamma: jax.Array,
    residual: jax.Array | None = None,
    *,
    block_k: int | None = None,
    eps: float = 1e-6,
    quant_max: float = 448.0,
) -> tuple[jax.Array, jax.Array]:
  """Reference un-fused multi-pass RMSNorm + FP8 Dynamic Quantization."""
  if residual is not None:
    x = x + residual

  x_f32 = x.astype(jnp.float32)
  var = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
  rsqrt_var = jax.lax.rsqrt(var + eps)
  gamma_f32 = gamma.astype(jnp.float32)
  x_norm = (x_f32 * rsqrt_var) * gamma_f32

  if block_k is None or block_k >= x_norm.shape[-1]:
    x_abs_max = jnp.max(jnp.abs(x_norm), axis=-1, keepdims=True)
    scale = jnp.maximum(x_abs_max / quant_max, 1e-12)
    x_scaled = x_norm / scale
    x_clamped = jnp.clip(x_scaled, -quant_max, quant_max)
    x_q = x_clamped.astype(jnp.float8_e4m3fn)
    scale_out = scale.astype(jnp.bfloat16)
  else:
    m, k = x_norm.shape
    num_blocks = k // block_k
    x_norm_blocked = jnp.reshape(x_norm, (m, num_blocks, block_k))
    x_abs_max = jnp.max(jnp.abs(x_norm_blocked), axis=-1, keepdims=True)
    scale = jnp.maximum(x_abs_max / quant_max, 1e-12)
    x_scaled = x_norm_blocked / scale
    x_clamped = jnp.clip(x_scaled, -quant_max, quant_max)
    x_q = jnp.reshape(x_clamped, (m, k)).astype(jnp.float8_e4m3fn)
    scale_out = jnp.reshape(scale, (m, num_blocks)).astype(jnp.bfloat16)

  return x_q, scale_out


class TestFusedRMSNormFP8Kernel:

  @pytest.mark.parametrize(
      "m,k,block_m,block_k,has_residual,return_residual",
      [
          (512, 5120, 256, None, False, False),
          (512, 5120, 256, None, True, True),
          (1024, 5120, 512, None, False, False),
          (1024, 5120, 512, None, True, True),
          (512, 5120, 256, 256, False, False),
      ],
  )
  def test_correctness_vs_reference(
      self, m, k, block_m, block_k, has_residual, return_residual
  ):
    key = jax.random.key(42)
    k_x, k_g, k_r = jax.random.split(key, 3)

    x = jax.random.normal(k_x, (m, k), dtype=jnp.bfloat16)
    gamma = jax.random.normal(k_g, (k,), dtype=jnp.bfloat16) * 0.1 + 1.0
    residual = (
        jax.random.normal(k_r, (m, k), dtype=jnp.bfloat16)
        if has_residual
        else None
    )

    # Reference
    ref_q, ref_s = reference_rmsnorm_fp8(x, gamma, residual, block_k=block_k)

    # Fused Pallas Kernel
    pallas_fn = jax.jit(
        lambda _x, _g, _r: fused_rmsnorm_fp8_quant(
            _x,
            _g,
            _r,
            block_m=block_m,
            block_k=block_k,
            return_residual=return_residual,
        )
    )
    pal_out = pallas_fn(x, gamma, residual)

    if return_residual and has_residual:
      pal_q, pal_s, pal_res = pal_out
      expected_res = x + residual
      res_diff = np.max(np.abs(np.array(pal_res) - np.array(expected_res)))
      assert res_diff < 1e-3, f"Residual mismatch: max diff = {res_diff}"
    else:
      pal_q, pal_s = pal_out

    # Dequantize both
    ref_q_f32 = ref_q.astype(jnp.float32)
    pal_q_f32 = pal_q.astype(jnp.float32)
    ref_s_f32 = ref_s.astype(jnp.float32)
    pal_s_f32 = pal_s.astype(jnp.float32)

    if block_k is None or block_k >= k:
      ref_dequant = ref_q_f32 * ref_s_f32
      pal_dequant = pal_q_f32 * pal_s_f32
    else:
      num_blocks = k // block_k
      ref_dequant = (
          jnp.reshape(ref_q_f32, (m, num_blocks, block_k))
          * jnp.expand_dims(ref_s_f32, -1)
      ).reshape((m, k))
      pal_dequant = (
          jnp.reshape(pal_q_f32, (m, num_blocks, block_k))
          * jnp.expand_dims(pal_s_f32, -1)
      ).reshape((m, k))

    ref_np = np.array(ref_dequant, dtype=np.float32)
    pal_np = np.array(pal_dequant, dtype=np.float32)

    cos_sim = np.dot(ref_np.flatten(), pal_np.flatten()) / (
        np.linalg.norm(ref_np.flatten()) * np.linalg.norm(pal_np.flatten())
        + 1e-12
    )
    assert (
        cos_sim > 0.99999
    ), f"Cosine similarity {cos_sim} is below required threshold 0.99999"

    # Scale check
    scale_diff = np.max(np.abs(np.array(ref_s_f32) - np.array(pal_s_f32)))
    assert scale_diff < 1e-4, f"Scale max diff {scale_diff} too large"
