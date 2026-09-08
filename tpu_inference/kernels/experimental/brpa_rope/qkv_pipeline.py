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
"""Pre-attention QKV GEMM, Head Norm, RoPE, and Zero-Copy Batched RPA with FP8 Numerics.

Implements FP8 (float8_e4m3fn) inputs and weights, BF16 GEMM outputs, BF16 Head Norm
and Vector RoPE, Zero-Copy RPAm handover, and dynamic FP8 quantization for the next GEMM
(matching Steps 1-16 of Qwen-32B prefill execution in Module 1005 / Module 1021).
"""

import jax
import jax.numpy as jnp
import numpy as np

from . import configs
from . import wrapper

FP8_DTYPE = jnp.float8_e4m3fn
FP8_MAX = 448.0


# ==============================================================================
# FP8 Quantization Utilities
# ==============================================================================
def quantize_to_fp8_dynamic(
    x: jax.Array,
    quant_max: float = FP8_MAX,
    dtype: jnp.dtype = FP8_DTYPE,
) -> tuple[jax.Array, jax.Array]:
  """Per-token (row-wise) dynamic FP8 quantization.

  Args:
    x: Activation tensor of shape [..., K] in bfloat16 / float32.
    quant_max: Maximum FP8 representable value (448.0 for float8_e4m3fn).
    dtype: Target FP8 dtype (jnp.float8_e4m3fn).

  Returns:
    x_q: Quantized activations in FP8 with identical shape to x.
    scale: Per-row scale factors of shape [..., 1] in bfloat16.
  """
  x_abs = jnp.abs(x.astype(jnp.float32))
  amax = jnp.max(x_abs, axis=-1, keepdims=True)
  scale = jnp.maximum(amax / quant_max, 1e-12).astype(jnp.bfloat16)
  x_scaled = x.astype(jnp.float32) / scale.astype(jnp.float32)
  x_clamped = jnp.clip(x_scaled, -quant_max, quant_max)
  x_q = x_clamped.astype(dtype)
  return x_q, scale


def quantize_weight_to_fp8_static(
    w: jax.Array,
    quant_max: float = FP8_MAX,
    dtype: jnp.dtype = FP8_DTYPE,
    axis: int = 0,
) -> tuple[jax.Array, jax.Array]:
  """Channel-wise static FP8 weight quantization along contracting dimension.

  Args:
    w: Weight tensor of shape [H_in, N] in bfloat16 / float32.
    quant_max: Maximum FP8 representable value (448.0 for float8_e4m3fn).
    dtype: Target FP8 dtype (jnp.float8_e4m3fn).
    axis: Contracting dimension for GEMM (default: 0).

  Returns:
    w_q: Quantized weight in FP8.
    scale: Per-channel scale factors of shape [1, N] in bfloat16.
  """
  w_abs = jnp.abs(w.astype(jnp.float32))
  amax = jnp.max(w_abs, axis=axis, keepdims=True)
  scale = jnp.maximum(amax / quant_max, 1e-12).astype(jnp.bfloat16)
  w_scaled = w.astype(jnp.float32) / scale.astype(jnp.float32)
  w_clamped = jnp.clip(w_scaled, -quant_max, quant_max)
  w_q = w_clamped.astype(dtype)
  return w_q, scale


def quantize_q_kv_major_weight_to_fp8(
    w_q_kv_major: jax.Array,
    quant_max: float = FP8_MAX,
    dtype: jnp.dtype = FP8_DTYPE,
) -> tuple[jax.Array, jax.Array]:
  """Channel-wise static FP8 quantization for KV-group major Q weight [N_kv, H_in, G*D]."""
  w_abs = jnp.abs(w_q_kv_major.astype(jnp.float32))
  amax = jnp.max(w_abs, axis=1, keepdims=True)  # along H_in
  scale = jnp.maximum(amax / quant_max, 1e-12).astype(jnp.bfloat16)
  w_scaled = w_q_kv_major.astype(jnp.float32) / scale.astype(jnp.float32)
  w_clamped = jnp.clip(w_scaled, -quant_max, quant_max)
  w_q = w_clamped.astype(dtype)
  return w_q, scale


# ==============================================================================
# Weight Permutation Utilities (Offline Transformation at Model Load)
# ==============================================================================
def permute_q_weight_to_kv_group_major(
    w_q_baseline: jax.Array,
    num_kv_heads: int,
    head_dim: int,
) -> jax.Array:
  """Permutes standard Q weight [H_in, N_q * head_dim] to [N_kv, H_in, G * head_dim]."""
  hidden_dim, total_q_dim = w_q_baseline.shape
  num_q_heads = total_q_dim // head_dim
  num_q_heads_per_kv_group = num_q_heads // num_kv_heads

  w_4d = w_q_baseline.reshape(
      hidden_dim, num_kv_heads, num_q_heads_per_kv_group, head_dim
  )
  w_kv_major = jnp.transpose(w_4d, (1, 0, 2, 3)).reshape(
      num_kv_heads, hidden_dim, num_q_heads_per_kv_group * head_dim
  )
  return w_kv_major


def permute_k_weight_to_kv_group_major(
    w_k_baseline: jax.Array,
    num_kv_heads: int,
    head_dim: int,
) -> jax.Array:
  """Permutes standard K weight [H_in, N_kv * head_dim] to [N_kv, H_in, head_dim]."""
  hidden_dim, _ = w_k_baseline.shape
  w_3d = w_k_baseline.reshape(hidden_dim, num_kv_heads, head_dim)
  return jnp.transpose(w_3d, (1, 0, 2))


def permute_v_weight_to_kv_group_major(
    w_v_baseline: jax.Array,
    num_kv_heads: int,
    head_dim: int,
) -> jax.Array:
  """Permutes standard V weight [H_in, N_kv * head_dim] to [N_kv, H_in, head_dim]."""
  hidden_dim, _ = w_v_baseline.shape
  w_3d = w_v_baseline.reshape(hidden_dim, num_kv_heads, head_dim)
  return jnp.transpose(w_3d, (1, 0, 2))


def permute_joint_kv_weights(
    w_k_baseline: jax.Array,
    w_v_baseline: jax.Array,
    num_kv_heads: int,
    head_dim: int,
) -> jax.Array:
  """Combines K and V weights into a single joint W_KV weight tensor [H_in, 2 * N_kv * head_dim].

  The output layout groups K (slice 0) and V (slice 1) along the KV slot dimension
  such that a single GEMM with X [T, H_in] produces KV [T, 2, N_kv, head_dim].

  Args:
    w_k_baseline: Baseline K weight of shape [H_in, N_kv * head_dim].
    w_v_baseline: Baseline V weight of shape [H_in, N_kv * head_dim].
    num_kv_heads: Number of key/value heads.
    head_dim: Dimension per attention head.

  Returns:
    w_kv_joint: Joint KV weight of shape [H_in, 2 * N_kv * head_dim].
  """
  hidden_dim, _ = w_k_baseline.shape
  w_k_4d = w_k_baseline.reshape(hidden_dim, 1, num_kv_heads, head_dim)
  w_v_4d = w_v_baseline.reshape(hidden_dim, 1, num_kv_heads, head_dim)
  w_kv_4d = jnp.concatenate([w_k_4d, w_v_4d], axis=1)
  return w_kv_4d.reshape(hidden_dim, 2 * num_kv_heads * head_dim)


def quantize_joint_kv_weight_to_fp8(
    w_kv_joint: jax.Array,
    quant_max: float = FP8_MAX,
    dtype: jnp.dtype = FP8_DTYPE,
) -> tuple[jax.Array, jax.Array]:
  """Per-channel static FP8 quantization for joint KV weight [H_in, 2 * N_kv * D]."""
  return quantize_weight_to_fp8_static(
      w_kv_joint, quant_max=quant_max, dtype=dtype, axis=0
  )


def permute_merged_qkv_weights(
    w_q_baseline: jax.Array,
    w_k_baseline: jax.Array,
    w_v_baseline: jax.Array,
    num_kv_heads: int,
    head_dim: int,
) -> jax.Array:
  """Interleaves Q, K, V weights by KV-head group into a single merged weight."""
  hidden_dim, total_q_dim = w_q_baseline.shape
  num_q_heads = total_q_dim // head_dim
  g = num_q_heads // num_kv_heads

  w_q_kv = permute_q_weight_to_kv_group_major(
      w_q_baseline, num_kv_heads, head_dim
  ).reshape(num_kv_heads, hidden_dim, g, head_dim)
  w_k_kv = permute_k_weight_to_kv_group_major(
      w_k_baseline, num_kv_heads, head_dim
  ).reshape(num_kv_heads, hidden_dim, 1, head_dim)
  w_v_kv = permute_v_weight_to_kv_group_major(
      w_v_baseline, num_kv_heads, head_dim
  ).reshape(num_kv_heads, hidden_dim, 1, head_dim)

  w_qkv_merged = jnp.concatenate([w_q_kv, w_k_kv, w_v_kv], axis=2).reshape(
      num_kv_heads, hidden_dim, (g + 2) * head_dim
  )
  return w_qkv_merged


# ==============================================================================
# Numerical Sub-Operations (Head RMSNorm & RoPE)
# ==============================================================================
def head_rms_norm(
    x: jax.Array,
    gamma: jax.Array,
    eps: float = 1e-6,
) -> jax.Array:
  """Per-head RMS normalization along the trailing head dimension (D=128)."""
  x_f32 = x.astype(jnp.float32)
  variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
  normed = x_f32 * jax.lax.rsqrt(variance + eps)
  scaled = normed * gamma.astype(jnp.float32)
  return scaled.astype(jnp.bfloat16)


def apply_reference_rope_5d(
    x: jax.Array,
    theta: float = 1000000.0,
    ordering: str = "split",
) -> jax.Array:
  """Applies Rotary Position Embedding to [N_kv, T, G, D] or [T, N_q, D]."""
  if x.ndim == 4:
    n_kv, total_tokens, g, d = x.shape
    half_d = d // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, d, 2, dtype=np.float32) / d))
    positions = np.arange(total_tokens, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)
    cos = np.cos(freqs)[None, :, None, :]
    sin = np.sin(freqs)[None, :, None, :]
    x_f32 = x.astype(jnp.float32)
    if ordering == "split":
      x1 = x_f32[..., :half_d]
      x2 = x_f32[..., half_d:]
      rot1 = x1 * cos - x2 * sin
      rot2 = x2 * cos + x1 * sin
      out = jnp.concatenate([rot1, rot2], axis=-1)
    else:
      x_pairs = x_f32.reshape(n_kv, total_tokens, g, half_d, 2)
      rot1 = x_pairs[..., 0] * cos - x_pairs[..., 1] * sin
      rot2 = x_pairs[..., 1] * cos + x_pairs[..., 0] * sin
      out = jnp.stack([rot1, rot2], axis=-1).reshape(n_kv, total_tokens, g, d)
    return out.astype(x.dtype)
  else:
    total_tokens, num_heads, d = x.shape
    half_d = d // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, d, 2, dtype=np.float32) / d))
    positions = np.arange(total_tokens, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)
    cos = np.cos(freqs)[:, None, :]
    sin = np.sin(freqs)[:, None, :]
    x_f32 = x.astype(jnp.float32)
    if ordering == "split":
      x1 = x_f32[..., :half_d]
      x2 = x_f32[..., half_d:]
      rot1 = x1 * cos - x2 * sin
      rot2 = x2 * cos + x1 * sin
      out = jnp.concatenate([rot1, rot2], axis=-1)
    else:
      x_pairs = x_f32.reshape(total_tokens, num_heads, half_d, 2)
      rot1 = x_pairs[..., 0] * cos - x_pairs[..., 1] * sin
      rot2 = x_pairs[..., 1] * cos + x_pairs[..., 0] * sin
      out = jnp.stack([rot1, rot2], axis=-1).reshape(total_tokens, num_heads, d)
    return out.astype(x.dtype)


# ==============================================================================
# Complete Pipelines (Steps 1 - 16)
# ==============================================================================
def qkv_projection_pipeline_kv_group_major(
    x_mid: jax.Array,
    w_q_kv_major: jax.Array,
    w_k_kv_major: jax.Array,
    w_v_kv_major: jax.Array,
    scale_w_q: jax.Array,
    scale_w_k: jax.Array,
    scale_w_v: jax.Array,
    gamma_input: jax.Array,
    gamma_q: jax.Array,
    gamma_k: jax.Array,
    num_kv_heads: int = 4,
    head_dim: int = 128,
    use_in_kernel_rope: bool = True,
    theta: float = 1000000.0,
    ordering: str = "split",
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Separated KV-Group Major QKV Pipeline."""
  # Step 1-3: Input RMSNorm & Dynamic FP8 Quantization
  x_norm = head_rms_norm(x_mid, gamma_input)
  x_fp8, scale_x = quantize_to_fp8_dynamic(x_norm)

  # Step 4-5: K Linear Projection & K Head RMSNorm
  k_dequant = (
      jnp.einsum("td,kdh->kth", x_fp8.astype(jnp.float32), w_k_kv_major.astype(jnp.float32))
      * scale_x.astype(jnp.float32)
      * scale_w_k.astype(jnp.float32)
  ).astype(jnp.bfloat16)
  k_norm = head_rms_norm(k_dequant, gamma_k)
  if not use_in_kernel_rope:
    k_rot = apply_reference_rope_5d(k_norm, theta=theta, ordering=ordering)
  else:
    k_rot = k_norm
  k_final = jnp.transpose(k_rot, (1, 0, 2)).astype(FP8_DTYPE)

  # Step 6: V Linear Projection
  v_dequant = (
      jnp.einsum("td,kdh->kth", x_fp8.astype(jnp.float32), w_v_kv_major.astype(jnp.float32))
      * scale_x.astype(jnp.float32)
      * scale_w_v.astype(jnp.float32)
  ).astype(jnp.bfloat16)
  v_final = jnp.transpose(v_dequant, (1, 0, 2)).astype(FP8_DTYPE)

  # Step 7-10: Q Linear Projection & Q Head RMSNorm
  q_dequant = (
      jnp.einsum("td,kdg->ktg", x_fp8.astype(jnp.float32), w_q_kv_major.astype(jnp.float32))
      * scale_x.astype(jnp.float32)
      * scale_w_q.astype(jnp.float32)
  ).astype(jnp.bfloat16)

  n_kv, total_tokens, gd = q_dequant.shape
  g = gd // head_dim
  q_4d = q_dequant.reshape(n_kv, total_tokens, g, head_dim)
  q_norm = head_rms_norm(q_4d, gamma_q)
  if not use_in_kernel_rope:
    q_final = apply_reference_rope_5d(q_norm, theta=theta, ordering=ordering)
  else:
    q_final = q_norm

  return q_final, k_final, v_final


def qkv_projection_pipeline_merged(
    x_mid: jax.Array,
    w_qkv_merged: jax.Array,
    scale_w_qkv: jax.Array,
    gamma_input: jax.Array,
    gamma_q: jax.Array,
    gamma_k: jax.Array,
    num_kv_heads: int = 4,
    head_dim: int = 128,
    use_in_kernel_rope: bool = True,
    theta: float = 1000000.0,
    ordering: str = "split",
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Merged Single-GEMM QKV Pipeline."""
  # Step 1-3: Input RMSNorm & Dynamic FP8 Quantization
  x_norm = head_rms_norm(x_mid, gamma_input)
  x_fp8, scale_x = quantize_to_fp8_dynamic(x_norm)

  # Single Merged FP8 GEMM
  qkv_dequant = (
      jnp.einsum("td,kdm->ktm", x_fp8.astype(jnp.float32), w_qkv_merged.astype(jnp.float32))
      * scale_x.astype(jnp.float32)
      * scale_w_qkv.astype(jnp.float32)
  ).astype(jnp.bfloat16)

  n_kv, total_tokens, total_m = qkv_dequant.shape
  total_heads = total_m // head_dim
  g = total_heads - 2

  q_raw = qkv_dequant[:, :, : g * head_dim].reshape(n_kv, total_tokens, g, head_dim)
  k_raw = qkv_dequant[:, :, g * head_dim : (g + 1) * head_dim].reshape(n_kv, total_tokens, head_dim)
  v_raw = qkv_dequant[:, :, (g + 1) * head_dim :].reshape(n_kv, total_tokens, head_dim)

  # Head RMSNorms
  q_norm = head_rms_norm(q_raw, gamma_q)
  if not use_in_kernel_rope:
    q_final = apply_reference_rope_5d(q_norm, theta=theta, ordering=ordering)
  else:
    q_final = q_norm

  k_norm = head_rms_norm(k_raw, gamma_k)
  if not use_in_kernel_rope:
    k_rot = apply_reference_rope_5d(k_norm, theta=theta, ordering=ordering)
  else:
    k_rot = k_norm
  k_final = jnp.transpose(k_rot, (1, 0, 2)).astype(FP8_DTYPE)

  v_final = jnp.transpose(v_raw, (1, 0, 2)).astype(FP8_DTYPE)

  return q_final, k_final, v_final


def qkv_projection_pipeline_baseline(
    x_mid: jax.Array,
    w_q_base: jax.Array,
    w_k_base: jax.Array,
    w_v_base: jax.Array,
    scale_w_q: jax.Array,
    scale_w_k: jax.Array,
    scale_w_v: jax.Array,
    gamma_input: jax.Array,
    gamma_q: jax.Array,
    gamma_k: jax.Array,
    num_kv_heads: int = 4,
    head_dim: int = 128,
    use_in_kernel_rope: bool = False,
    theta: float = 1000000.0,
    ordering: str = "split",
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Standard Token-Major Baseline Pipeline."""
  # Step 1-3: Input RMSNorm & Dynamic FP8 Quantization
  x_norm = head_rms_norm(x_mid, gamma_input)
  x_fp8, scale_x = quantize_to_fp8_dynamic(x_norm)

  # Step 4-5: K GEMM & Norm
  k_dequant = (
      jnp.dot(x_fp8.astype(jnp.float32), w_k_base.astype(jnp.float32))
      * scale_x.astype(jnp.float32)
      * scale_w_k.astype(jnp.float32)
  ).astype(jnp.bfloat16)
  total_tokens, _ = x_mid.shape
  k_3d = k_dequant.reshape(total_tokens, num_kv_heads, head_dim)
  k_norm = head_rms_norm(k_3d, gamma_k)
  if not use_in_kernel_rope:
    k_final = apply_reference_rope_5d(k_norm, theta=theta, ordering=ordering).astype(FP8_DTYPE)
  else:
    k_final = k_norm.astype(FP8_DTYPE)

  # Step 6: V GEMM
  v_dequant = (
      jnp.dot(x_fp8.astype(jnp.float32), w_v_base.astype(jnp.float32))
      * scale_x.astype(jnp.float32)
      * scale_w_v.astype(jnp.float32)
  ).astype(jnp.bfloat16)
  v_final = v_dequant.reshape(total_tokens, num_kv_heads, head_dim).astype(FP8_DTYPE)

  # Step 7-10: Q GEMM & Norm
  q_dequant = (
      jnp.dot(x_fp8.astype(jnp.float32), w_q_base.astype(jnp.float32))
      * scale_x.astype(jnp.float32)
      * scale_w_q.astype(jnp.float32)
  ).astype(jnp.bfloat16)
  num_q_heads = q_dequant.shape[-1] // head_dim
  q_3d = q_dequant.reshape(total_tokens, num_q_heads, head_dim)
  q_norm = head_rms_norm(q_3d, gamma_q)
  if not use_in_kernel_rope:
    q_final = apply_reference_rope_5d(q_norm, theta=theta, ordering=ordering)
  else:
    q_final = q_norm

  return q_final, k_final, v_final


def qkv_projection_pipeline_head_major_q_joint_kv(
    x_mid: jax.Array,
    w_q_kv_major: jax.Array,
    w_kv_joint: jax.Array,
    scale_w_q: jax.Array,
    scale_w_kv: jax.Array,
    gamma_input: jax.Array,
    gamma_q: jax.Array,
    gamma_k: jax.Array,
    num_kv_heads: int = 4,
    head_dim: int = 128,
    use_in_kernel_rope: bool = True,
    theta: float = 1000000.0,
    ordering: str = "split",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Decoupled Head-Major Q GEMM + Joint W_KV GEMM Pipeline.

  1. Input RMSNorm + Dynamic FP8 Quantization:
     x_mid [T, H_in] -> x_fp8 [T, H_in] (FP8), scale_x [T, 1] (BF16)

  2. Dedicated Head-Major Q GEMM:
     x_fp8 [T, H_in] x w_q [N_kv, H_in, G*D] -> q_dequant [N_kv, T, G*D] -> [N_kv, T, G, D]
     Directly outputs head-major format [4, 4096, 8, 128], eliminating Step 11 layout copy!

  3. Joint W_KV GEMM:
     x_fp8 [T, H_in] x w_kv [H_in, 2*N_kv*D] -> kv_dequant [T, 2, N_kv, D]
     Single joint GEMM for K and V, saving compute latency and intermediate memory copies.

  4. Fused Vector Post-Processing:
     - Head RMSNorm on Q [N_kv, T, G, D] (and RoPE if not in-kernel).
     - Head RMSNorm on K slice [T, N_kv, D] (and RoPE if not in-kernel).
     - V slice [T, N_kv, D] bypasses RoPE.
     - Quantize K and V to FP8.
     - Stack/format into joint KV cache tensor [T, 2, N_kv, D].

  Args:
    x_mid: Input residual activations [T, H_in] in BF16.
    w_q_kv_major: Offline permuted Q weight [N_kv, H_in, G * D] in FP8.
    w_kv_joint: Offline permuted joint KV weight [H_in, 2 * N_kv * D] in FP8.
    scale_w_q: Static per-channel weight scales for Q [N_kv, 1, G * D] in BF16.
    scale_w_kv: Static per-channel weight scales for KV [1, 2 * N_kv * D] in BF16.
    gamma_input: Input RMSNorm scale parameters [H_in] in BF16.
    gamma_q: Q Head RMSNorm scale parameters [D] in BF16.
    gamma_k: K Head RMSNorm scale parameters [D] in BF16.
    num_kv_heads: Number of key/value heads (e.g. 4 for TP=2 shard).
    head_dim: Attention head dimension (e.g. 128).
    use_in_kernel_rope: If True, defer RoPE to in-kernel RPAm computation.
    theta: RoPE base frequency.
    ordering: RoPE layout ('split' or 'interleaved').

  Returns:
    q_final: Head-major Query tensor [N_kv, T, G, D] in BF16.
    k_final: Key tensor [T, N_kv, D] in FP8.
    v_final: Value tensor [T, N_kv, D] in FP8.
    kv_final: Joint Key/Value tensor [T, 2, N_kv, D] in FP8.
  """
  # Step 1-3: Input RMSNorm & Dynamic FP8 Quantization
  x_norm = head_rms_norm(x_mid, gamma_input)
  x_fp8, scale_x = quantize_to_fp8_dynamic(x_norm)

  # Step 4: Head-Major Q GEMM
  q_dequant = (
      jnp.einsum(
          "td,kdg->ktg",
          x_fp8.astype(jnp.float32),
          w_q_kv_major.astype(jnp.float32),
      )
      * scale_x.astype(jnp.float32)
      * scale_w_q.astype(jnp.float32)
  ).astype(jnp.bfloat16)

  n_kv, total_tokens, gd = q_dequant.shape
  g = gd // head_dim
  q_4d = q_dequant.reshape(n_kv, total_tokens, g, head_dim)
  q_norm = head_rms_norm(q_4d, gamma_q)
  if not use_in_kernel_rope:
    q_final = apply_reference_rope_5d(q_norm, theta=theta, ordering=ordering)
  else:
    q_final = q_norm

  # Step 5: Joint W_KV GEMM
  kv_dequant = (
      jnp.dot(x_fp8.astype(jnp.float32), w_kv_joint.astype(jnp.float32))
      * scale_x.astype(jnp.float32)
      * scale_w_kv.astype(jnp.float32)
  ).astype(jnp.bfloat16)
  kv_4d = kv_dequant.reshape(total_tokens, 2, num_kv_heads, head_dim)

  # Step 6: Vector Post-Processing on K and V
  k_raw = kv_4d[:, 0, :, :]
  v_raw = kv_4d[:, 1, :, :]

  # RMSNorm on K only
  k_norm = head_rms_norm(k_raw, gamma_k)
  if not use_in_kernel_rope:
    k_rot = apply_reference_rope_5d(k_norm, theta=theta, ordering=ordering)
  else:
    k_rot = k_norm
  k_final = k_rot.astype(FP8_DTYPE)

  # V bypasses RMSNorm and RoPE
  v_final = v_raw.astype(FP8_DTYPE)

  # Joint KV output tensor directly matching Paged KV Cache block layout
  kv_final = jnp.stack([k_final, v_final], axis=1)

  return q_final, k_final, v_final, kv_final

