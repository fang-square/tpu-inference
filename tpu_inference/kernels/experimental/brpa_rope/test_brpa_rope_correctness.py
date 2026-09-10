"""Numerical parity test for bRPA with In-Kernel RoPE."""

import os
import sys
from absl import app
from absl import logging
import jax
import jax.numpy as jnp
import numpy as np

try:
  from tpu_inference.kernels.experimental.brpa_rope import configs, pallas_rmsnorm, qkv_pipeline, utils, wrapper
except (ModuleNotFoundError, ImportError):
  try:
    from google3.experimental.users.fangfangz.kernels.brpa_rope import configs, pallas_rmsnorm, qkv_pipeline, utils, wrapper
  except ModuleNotFoundError:
    from experimental.users.fangfangz.kernels.brpa_rope import configs, pallas_rmsnorm, qkv_pipeline, utils, wrapper


def apply_reference_rope(
    x: jax.Array, theta: float = 1000000.0, ordering: str = "split"
) -> jax.Array:
  """Standard Reference RoPE on [T, H, D]."""
  T, H, D = x.shape
  half_D = D // 2
  inv_freq = 1.0 / (theta ** (np.arange(0, D, 2, dtype=np.float32) / D))
  positions = np.arange(T, dtype=np.float32)
  freqs = np.outer(positions, inv_freq)
  cos = np.cos(freqs)[:, None, :]  # [T, 1, half_D]
  sin = np.sin(freqs)[:, None, :]  # [T, 1, half_D]

  x_f32 = x.astype(jnp.float32)
  if ordering == "split":
    x1 = x_f32[..., :half_D]
    x2 = x_f32[..., half_D:]
    rot1 = x1 * cos - x2 * sin
    rot2 = x2 * cos + x1 * sin
    out = jnp.concatenate([rot1, rot2], axis=-1)
  else:
    x_pairs = x_f32.reshape(T, H, half_D, 2)
    rot1 = x_pairs[..., 0] * cos - x_pairs[..., 1] * sin
    rot2 = x_pairs[..., 1] * cos + x_pairs[..., 0] * sin
    out = jnp.stack([rot1, rot2], axis=-1).reshape(T, H, D)
  return out.astype(x.dtype)


def reference_attention(
    q_rot: jax.Array, k_rot: jax.Array, v: jax.Array, sm_scale: float
) -> jax.Array:
  """Reference Causal Multi-Head / Grouped-Query Attention on [T, H, D]."""
  T, H_q, D = q_rot.shape
  _, H_kv, _ = k_rot.shape
  G = H_q // H_kv

  # Repeat KV heads for GQA
  k_rep = jnp.repeat(k_rot, G, axis=1)  # [T, H_q, D]
  v_rep = jnp.repeat(v, G, axis=1)  # [T, H_q, D]

  # [H_q, T, D]
  q_t = jnp.transpose(q_rot, (1, 0, 2))
  k_t = jnp.transpose(k_rep, (1, 0, 2))
  v_t = jnp.transpose(v_rep, (1, 0, 2))

  # Scores: [H_q, T, T]
  scores = jnp.matmul(q_t, jnp.transpose(k_t, (0, 2, 1))) * sm_scale

  # Causal mask
  mask = jnp.tril(jnp.ones((T, T), dtype=bool))
  scores = jnp.where(mask[None, :, :], scores, -1e9)

  # Softmax & Attn Out
  probs = jax.nn.softmax(scores, axis=-1)
  out = jnp.matmul(probs, v_t)  # [H_q, T, D]
  return jnp.transpose(out, (1, 0, 2)).astype(q_rot.dtype)


def test_in_kernel_rope_correctness():
  print("=" * 80)
  print("TEST 1: IN-KERNEL RoPE NUMERICAL PARITY (TOKEN-MAJOR)")
  print("=" * 80)

  T = 128
  H_q = 8
  H_kv = 2
  D = 128
  theta = 1000000.0
  sm_scale = 1.0 / np.sqrt(D)
  dtype = jnp.float32

  key = jax.random.PRNGKey(42)
  k1, k2, k3 = jax.random.split(key, 3)

  q_raw = jax.random.normal(k1, (T, H_q, D), dtype=dtype)
  k_raw = jax.random.normal(k2, (T, H_kv, D), dtype=dtype)
  v_raw = jax.random.normal(k3, (T, H_kv, D), dtype=dtype)

  q_rot_ref = apply_reference_rope(q_raw, theta=theta, ordering="split")
  k_rot_ref = apply_reference_rope(k_raw, theta=theta, ordering="split")
  attn_ref = reference_attention(q_rot_ref, k_rot_ref, v_raw, sm_scale=sm_scale)

  page_size = 16
  num_pages = (T + page_size - 1) // page_size + 4
  kv_cache = jnp.zeros(
      wrapper.get_kv_cache_shape(
          total_num_pages=num_pages,
          page_size=page_size,
          actual_num_kv_heads=H_kv,
          actual_head_dim=D,
          kv_dtype=dtype,
          kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
      ),
      dtype=dtype,
  )
  kv_lens = jnp.array([T], dtype=jnp.int32)
  page_indices = jnp.arange(num_pages, dtype=jnp.int32)
  cu_q_lens = jnp.array([0, T], dtype=jnp.int32)
  distribution = jnp.array([0, 1, 1], dtype=jnp.int32)

  attn_brpa, _ = wrapper.ragged_paged_attention_rope(
      queries=q_raw,
      keys=k_raw,
      values=v_raw,
      kv_cache=kv_cache,
      kv_lens=kv_lens,
      page_indices=page_indices,
      cu_q_lens=cu_q_lens,
      distribution=distribution,
      rope_theta=theta,
      rope_dim=D,
      rope_input_ordering="split",
      sm_scale=sm_scale,
      kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
      is_kv_group_major=False,
  )

  diff = jnp.abs(attn_brpa - attn_ref)
  max_diff = float(jnp.max(diff))
  mean_diff = float(jnp.mean(diff))
  print(
      f"  Token-Major vs Ref: Max diff: {max_diff:.6e}, Mean diff:"
      f" {mean_diff:.6e}"
  )
  assert max_diff < 1e-4, f"Max diff {max_diff} exceeded tolerance"
  print(">>> TEST 1 PASSED <<<")


def test_kv_group_major_correctness():
  print("=" * 80)
  print("TEST 2: KV-GROUP MAJOR Q ZERO-COPY + IN-KERNEL RoPE")
  print("=" * 80)

  T = 128
  H_q = 8
  H_kv = 2
  G = H_q // H_kv  # 4
  D = 128
  theta = 1000000.0
  sm_scale = 1.0 / np.sqrt(D)
  dtype = jnp.float32

  key = jax.random.PRNGKey(42)
  k1, k2, k3 = jax.random.split(key, 3)

  q_raw_token_major = jax.random.normal(k1, (T, H_q, D), dtype=dtype)
  k_raw = jax.random.normal(k2, (T, H_kv, D), dtype=dtype)
  v_raw = jax.random.normal(k3, (T, H_kv, D), dtype=dtype)

  # Permute Q to KV-Group Major: [H_kv, T, G, D]
  q_kv_major = q_raw_token_major.reshape(T, H_kv, G, D).transpose(1, 0, 2, 3)

  # Reference attention output
  q_rot_ref = apply_reference_rope(
      q_raw_token_major, theta=theta, ordering="split"
  )
  k_rot_ref = apply_reference_rope(k_raw, theta=theta, ordering="split")
  attn_ref = reference_attention(q_rot_ref, k_rot_ref, v_raw, sm_scale=sm_scale)

  page_size = 16
  num_pages = (T + page_size - 1) // page_size + 4
  kv_cache = jnp.zeros(
      wrapper.get_kv_cache_shape(
          total_num_pages=num_pages,
          page_size=page_size,
          actual_num_kv_heads=H_kv,
          actual_head_dim=D,
          kv_dtype=dtype,
          kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
      ),
      dtype=dtype,
  )
  kv_lens = jnp.array([T], dtype=jnp.int32)
  page_indices = jnp.arange(num_pages, dtype=jnp.int32)
  cu_q_lens = jnp.array([0, T], dtype=jnp.int32)
  distribution = jnp.array([0, 1, 1], dtype=jnp.int32)

  attn_brpa_kv, _ = wrapper.ragged_paged_attention_rope(
      queries=q_kv_major,
      keys=k_raw,
      values=v_raw,
      kv_cache=kv_cache,
      kv_lens=kv_lens,
      page_indices=page_indices,
      cu_q_lens=cu_q_lens,
      distribution=distribution,
      rope_theta=theta,
      rope_dim=D,
      rope_input_ordering="split",
      sm_scale=sm_scale,
      kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
      is_kv_group_major=True,
  )

  # attn_brpa_kv is [H_kv, T, G, D], reshape to [T, H_q, D] for comparison
  attn_brpa_reconstructed = attn_brpa_kv.transpose(1, 0, 2, 3).reshape(
      T, H_q, D
  )

  diff = jnp.abs(attn_brpa_reconstructed - attn_ref)
  max_diff = float(jnp.max(diff))
  mean_diff = float(jnp.mean(diff))
  print(
      f"  KV-Group Major vs Ref: Max diff: {max_diff:.6e}, Mean diff:"
      f" {mean_diff:.6e}"
  )
  assert max_diff < 1e-4, f"Max diff {max_diff} exceeded tolerance"
  print(">>> TEST 2 PASSED <<<")


def test_qkv_pipeline_equivalence():
  print("=" * 80)
  print("TEST 3: SEPARATED vs MERGED QKV PIPELINE EQUIVALENCE")
  print("=" * 80)

  T = 64
  H_in = 256
  H_q = 8
  H_kv = 2
  D = 128
  G = H_q // H_kv

  key = jax.random.PRNGKey(123)
  k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

  x_mid = jax.random.normal(k1, (T, H_in), dtype=jnp.bfloat16)
  w_q = jax.random.normal(k2, (H_in, H_q * D), dtype=jnp.bfloat16) * 0.02
  w_k = jax.random.normal(k3, (H_in, H_kv * D), dtype=jnp.bfloat16) * 0.02
  w_v = jax.random.normal(k4, (H_in, H_kv * D), dtype=jnp.bfloat16) * 0.02

  gamma_in = jnp.ones((H_in,), dtype=jnp.bfloat16)
  gamma_q = jnp.ones((D,), dtype=jnp.bfloat16)
  gamma_k = jnp.ones((D,), dtype=jnp.bfloat16)

  # Permute weights offline
  w_q_kv = qkv_pipeline.permute_q_weight_to_kv_group_major(w_q, H_kv, D)
  w_k_kv = qkv_pipeline.permute_k_weight_to_kv_group_major(w_k, H_kv, D)
  w_v_kv = qkv_pipeline.permute_v_weight_to_kv_group_major(w_v, H_kv, D)

  w_qkv_merged = qkv_pipeline.permute_merged_qkv_weights(w_q, w_k, w_v, H_kv, D)

  # Quantize weights statically
  w_q_fp8, scale_w_q = qkv_pipeline.quantize_q_kv_major_weight_to_fp8(w_q_kv)
  w_k_fp8, scale_w_k = qkv_pipeline.quantize_q_kv_major_weight_to_fp8(w_k_kv)
  w_v_fp8, scale_w_v = qkv_pipeline.quantize_q_kv_major_weight_to_fp8(w_v_kv)
  w_qkv_fp8, scale_w_qkv = qkv_pipeline.quantize_q_kv_major_weight_to_fp8(
      w_qkv_merged
  )

  # Run Separated Pipeline
  q_sep, k_sep, v_sep = qkv_pipeline.qkv_projection_pipeline_kv_group_major(
      x_mid=x_mid,
      w_q_kv_major=w_q_fp8,
      w_k_kv_major=w_k_fp8,
      w_v_kv_major=w_v_fp8,
      scale_w_q=scale_w_q,
      scale_w_k=scale_w_k,
      scale_w_v=scale_w_v,
      gamma_input=gamma_in,
      gamma_q=gamma_q,
      gamma_k=gamma_k,
      num_kv_heads=H_kv,
      head_dim=D,
      use_in_kernel_rope=True,
  )

  # Run Merged Pipeline
  q_mrg, k_mrg, v_mrg = qkv_pipeline.qkv_projection_pipeline_merged(
      x_mid=x_mid,
      w_qkv_merged=w_qkv_fp8,
      scale_w_qkv=scale_w_qkv,
      gamma_input=gamma_in,
      gamma_q=gamma_q,
      gamma_k=gamma_k,
      num_kv_heads=H_kv,
      head_dim=D,
      use_in_kernel_rope=True,
  )

  cos_sim_q = float(
      jnp.sum(q_sep.astype(jnp.float32) * q_mrg.astype(jnp.float32))
      / (
          jnp.linalg.norm(q_sep.astype(jnp.float32))
          * jnp.linalg.norm(q_mrg.astype(jnp.float32))
      )
  )
  print(f"  Separated vs Merged Q Cosine Similarity: {cos_sim_q:.6f}")
  assert cos_sim_q > 0.999, f"Q cosine similarity {cos_sim_q} too low"
  print(">>> TEST 3 PASSED <<<")


def test_strided_dma_correctness():
  print("=" * 80)
  print("TEST 4: TOKEN-MAJOR STRIDED DMA ZERO-COPY + IN-KERNEL RoPE")
  print("=" * 80)

  T = 128
  H_q = 8
  H_kv = 2
  D = 128
  theta = 1000000.0
  sm_scale = 1.0 / np.sqrt(D)
  dtype = jnp.float32

  key = jax.random.PRNGKey(42)
  k1, k2, k3 = jax.random.split(key, 3)

  q_raw_token_major = jax.random.normal(k1, (T, H_q, D), dtype=dtype)
  k_raw = jax.random.normal(k2, (T, H_kv, D), dtype=dtype)
  v_raw = jax.random.normal(k3, (T, H_kv, D), dtype=dtype)

  # Reference attention output
  q_rot_ref = apply_reference_rope(
      q_raw_token_major, theta=theta, ordering="split"
  )
  k_rot_ref = apply_reference_rope(k_raw, theta=theta, ordering="split")
  attn_ref = reference_attention(q_rot_ref, k_rot_ref, v_raw, sm_scale=sm_scale)

  page_size = 16
  num_pages = (T + page_size - 1) // page_size + 4
  kv_cache = jnp.zeros(
      wrapper.get_kv_cache_shape(
          total_num_pages=num_pages,
          page_size=page_size,
          actual_num_kv_heads=H_kv,
          actual_head_dim=D,
          kv_dtype=dtype,
          kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
      ),
      dtype=dtype,
  )
  kv_lens = jnp.array([T], dtype=jnp.int32)
  page_indices = jnp.arange(num_pages, dtype=jnp.int32)
  cu_q_lens = jnp.array([0, T], dtype=jnp.int32)
  distribution = jnp.array([0, 1, 1], dtype=jnp.int32)

  attn_brpa_strided, _ = wrapper.ragged_paged_attention_rope(
      queries=q_raw_token_major,
      keys=k_raw,
      values=v_raw,
      kv_cache=kv_cache,
      kv_lens=kv_lens,
      page_indices=page_indices,
      cu_q_lens=cu_q_lens,
      distribution=distribution,
      rope_theta=theta,
      rope_dim=D,
      rope_input_ordering="split",
      sm_scale=sm_scale,
      kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
      is_kv_group_major=False,
      use_strided_dma=True,
  )

  diff = jnp.abs(attn_brpa_strided - attn_ref)
  max_diff = float(jnp.max(diff))
  mean_diff = float(jnp.mean(diff))
  print(
      f"  Strided DMA vs Ref: Max diff: {max_diff:.6e}, Mean diff:"
      f" {mean_diff:.6e}"
  )
  assert max_diff < 1e-4, f"Max diff {max_diff} exceeded tolerance"
  print(">>> TEST 4 PASSED <<<")


def test_head_major_q_joint_kv_correctness():
  print("=" * 80)
  print("TEST 5: DECOUPLED HEAD-MAJOR Q + JOINT KV PIPELINE EQUIVALENCE")
  print("=" * 80)

  T = 64
  H_in = 256
  H_q = 8
  H_kv = 2
  D = 128
  G = H_q // H_kv  # 4

  key = jax.random.PRNGKey(42)
  k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

  x_mid = jax.random.normal(k1, (T, H_in), dtype=jnp.bfloat16)
  w_q = jax.random.normal(k2, (H_in, H_q * D), dtype=jnp.bfloat16) * 0.02
  w_k = jax.random.normal(k3, (H_in, H_kv * D), dtype=jnp.bfloat16) * 0.02
  w_v = jax.random.normal(k4, (H_in, H_kv * D), dtype=jnp.bfloat16) * 0.02

  gamma_in = jnp.ones((H_in,), dtype=jnp.bfloat16)
  gamma_q = jnp.ones((D,), dtype=jnp.bfloat16)
  gamma_k = jnp.ones((D,), dtype=jnp.bfloat16)

  # Permute weights offline
  w_q_kv = qkv_pipeline.permute_q_weight_to_kv_group_major(w_q, H_kv, D)
  w_kv_joint = qkv_pipeline.permute_joint_kv_weights(w_k, w_v, H_kv, D)

  # Quantize weights statically to FP8
  w_q_fp8, scale_w_q = qkv_pipeline.quantize_q_kv_major_weight_to_fp8(w_q_kv)
  w_kv_fp8, scale_w_kv = qkv_pipeline.quantize_joint_kv_weight_to_fp8(
      w_kv_joint
  )
  w_k_fp8, scale_w_k = qkv_pipeline.quantize_weight_to_fp8_static(w_k)
  w_v_fp8, scale_w_v = qkv_pipeline.quantize_weight_to_fp8_static(w_v)
  w_q_base_fp8, scale_w_q_base = qkv_pipeline.quantize_weight_to_fp8_static(w_q)

  # Run Baseline Pipeline (with reference RoPE)
  q_base, k_base, v_base = qkv_pipeline.qkv_projection_pipeline_baseline(
      x_mid=x_mid,
      w_q_base=w_q_base_fp8,
      w_k_base=w_k_fp8,
      w_v_base=w_v_fp8,
      scale_w_q=scale_w_q_base,
      scale_w_k=scale_w_k,
      scale_w_v=scale_w_v,
      gamma_input=gamma_in,
      gamma_q=gamma_q,
      gamma_k=gamma_k,
      num_kv_heads=H_kv,
      head_dim=D,
      use_in_kernel_rope=False,
  )

  # Run Decoupled Head-Major Q + Joint KV Pipeline (with reference RoPE)
  q_new, k_new, v_new, kv_new = (
      qkv_pipeline.qkv_projection_pipeline_head_major_q_joint_kv(
          x_mid=x_mid,
          w_q_kv_major=w_q_fp8,
          w_kv_joint=w_kv_fp8,
          scale_w_q=scale_w_q,
          scale_w_kv=scale_w_kv,
          gamma_input=gamma_in,
          gamma_q=gamma_q,
          gamma_k=gamma_k,
          num_kv_heads=H_kv,
          head_dim=D,
          use_in_kernel_rope=False,
      )
  )

  # 1. Verify Q numerical similarity (reshape head-major [H_kv, T, G, D] -> [T, H_q, D])
  q_new_flat = q_new.transpose(1, 0, 2, 3).reshape(T, H_q, D)
  cos_sim_q = float(
      jnp.sum(q_base.astype(jnp.float32) * q_new_flat.astype(jnp.float32))
      / (
          jnp.linalg.norm(q_base.astype(jnp.float32))
          * jnp.linalg.norm(q_new_flat.astype(jnp.float32))
      )
  )
  print(f"  Head-Major Q vs Baseline Q Cosine Similarity:  {cos_sim_q:.6f}")
  assert cos_sim_q > 0.999, f"Q cosine similarity {cos_sim_q} too low"

  # 2. Verify K numerical similarity
  cos_sim_k = float(
      jnp.sum(k_base.astype(jnp.float32) * k_new.astype(jnp.float32))
      / (
          jnp.linalg.norm(k_base.astype(jnp.float32))
          * jnp.linalg.norm(k_new.astype(jnp.float32))
      )
  )
  print(f"  Joint KV (K slice) vs Baseline K Cosine Sim:   {cos_sim_k:.6f}")
  assert cos_sim_k > 0.999, f"K cosine similarity {cos_sim_k} too low"

  # 3. Verify V numerical similarity
  cos_sim_v = float(
      jnp.sum(v_base.astype(jnp.float32) * v_new.astype(jnp.float32))
      / (
          jnp.linalg.norm(v_base.astype(jnp.float32))
          * jnp.linalg.norm(v_new.astype(jnp.float32))
      )
  )
  print(f"  Joint KV (V slice) vs Baseline V Cosine Sim:   {cos_sim_v:.6f}")
  assert cos_sim_v > 0.999, f"V cosine similarity {cos_sim_v} too low"

  # 4. Verify Joint KV tensor exact slice identity
  diff_k = jnp.max(
      jnp.abs(
          kv_new[:, 0, :, :].astype(jnp.float32) - k_new.astype(jnp.float32)
      )
  )
  diff_v = jnp.max(
      jnp.abs(
          kv_new[:, 1, :, :].astype(jnp.float32) - v_new.astype(jnp.float32)
      )
  )
  print(f"  KV Joint Slice 0 vs K Max Absolute Diff:       {float(diff_k):.6e}")
  print(f"  KV Joint Slice 1 vs V Max Absolute Diff:       {float(diff_v):.6e}")
  assert diff_k == 0.0, "KV tensor slice 0 does not match K"
  assert diff_v == 0.0, "KV tensor slice 1 does not match V"

  print(">>> TEST 5 PASSED <<<")


def test_in_kernel_norm_rope_tpu_correctness():
  print("=" * 80)
  print("TEST 6: IN-KERNEL RMSNORM + ROPE NUMERICAL PARITY (TPU)")
  print("=" * 80)

  T = 128
  H_q = 8
  H_kv = 2
  G = H_q // H_kv
  D = 128
  theta = 1000000.0
  sm_scale = 1.0 / np.sqrt(D)
  dtype = jnp.float32

  key = jax.random.PRNGKey(42)
  k1, k2, k3, k4, k5 = jax.random.split(key, 5)

  q_raw_token_major = jax.random.normal(k1, (T, H_q, D), dtype=dtype)
  k_raw = jax.random.normal(k2, (T, H_kv, D), dtype=dtype)
  v_raw = jax.random.normal(k3, (T, H_kv, D), dtype=dtype)
  gamma_q_arr = jnp.ones((D,), dtype=dtype)
  gamma_k_arr = jnp.ones((D,), dtype=dtype)
  gamma_q = None
  gamma_k = None

  # Reference path: separate RMSNorm followed by RoPE
  q_norm_ref = qkv_pipeline.head_rms_norm(q_raw_token_major, gamma_q_arr)
  k_norm_ref = qkv_pipeline.head_rms_norm(k_raw, gamma_k_arr)
  q_rot_ref = apply_reference_rope(q_norm_ref, theta=theta, ordering="split")
  k_rot_ref = apply_reference_rope(k_norm_ref, theta=theta, ordering="split")
  attn_ref = reference_attention(q_rot_ref, k_rot_ref, v_raw, sm_scale=sm_scale)

  page_size = 16
  num_pages = (T + page_size - 1) // page_size + 4
  kv_cache = jnp.zeros(
      wrapper.get_kv_cache_shape(
          total_num_pages=num_pages,
          page_size=page_size,
          actual_num_kv_heads=H_kv,
          actual_head_dim=D,
          kv_dtype=dtype,
          kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
      ),
      dtype=dtype,
  )
  kv_lens = jnp.array([T], dtype=jnp.int32)
  page_indices = jnp.arange(num_pages, dtype=jnp.int32)
  cu_q_lens = jnp.array([0, T], dtype=jnp.int32)
  distribution = jnp.array([0, 1, 1], dtype=jnp.int32)

  # 1. Test 1 Configuration: KV-Group Major Q + In-Kernel Norm & RoPE
  q_kv_major = q_raw_token_major.reshape(T, H_kv, G, D).transpose(1, 0, 2, 3)
  attn_test1, _ = wrapper.ragged_paged_attention_rope(
      queries=q_kv_major,
      keys=k_norm_ref,
      values=v_raw,
      kv_cache=kv_cache,
      kv_lens=kv_lens,
      page_indices=page_indices,
      cu_q_lens=cu_q_lens,
      distribution=distribution,
      rope_theta=theta,
      rope_dim=D,
      rope_input_ordering="split",
      sm_scale=sm_scale,
      kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
      is_kv_group_major=True,
      use_strided_dma=False,
      apply_rmsnorm=True,
      gamma_q=gamma_q,
  )
  attn_test1_flat = attn_test1.transpose(1, 0, 2, 3).reshape(T, H_q, D)
  diff_1 = jnp.abs(attn_test1_flat - attn_ref)
  max_diff_1 = float(jnp.max(diff_1))
  print(
      f"  Test 1 (KV-Major + In-Kernel Norm & RoPE) Max Diff: {max_diff_1:.6e}"
  )
  assert max_diff_1 < 1e-4, f"Test 1 diff {max_diff_1} exceeded tolerance"

  # 2. Test 2 Configuration: Token Major Q + Separate Q-Norm + Strided DMA + In-Kernel RoPE
  attn_test2, _ = wrapper.ragged_paged_attention_rope(
      queries=q_norm_ref,
      keys=k_norm_ref,
      values=v_raw,
      kv_cache=kv_cache,
      kv_lens=kv_lens,
      page_indices=page_indices,
      cu_q_lens=cu_q_lens,
      distribution=distribution,
      rope_theta=theta,
      rope_dim=D,
      rope_input_ordering="split",
      sm_scale=sm_scale,
      kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
      is_kv_group_major=False,
      use_strided_dma=True,
      apply_rmsnorm=False,
  )
  diff_2 = jnp.abs(attn_test2 - attn_ref)
  max_diff_2 = float(jnp.max(diff_2))
  print(
      "  Test 2 (Token Major + Separate Norm + Strided DMA) Max Diff:"
      f" {max_diff_2:.6e}"
  )
  assert max_diff_2 < 1e-4, f"Test 2 diff {max_diff_2} exceeded tolerance"

  # 3. Test 3 Configuration: KV-Major Q + Strided DMA + In-Kernel Norm & RoPE
  attn_test3, _ = wrapper.ragged_paged_attention_rope(
      queries=q_kv_major,
      keys=k_norm_ref,
      values=v_raw,
      kv_cache=kv_cache,
      kv_lens=kv_lens,
      page_indices=page_indices,
      cu_q_lens=cu_q_lens,
      distribution=distribution,
      rope_theta=theta,
      rope_dim=D,
      rope_input_ordering="split",
      sm_scale=sm_scale,
      kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
      is_kv_group_major=True,
      use_strided_dma=True,
      apply_rmsnorm=True,
      gamma_q=gamma_q,
  )
  attn_test3_flat = attn_test3.transpose(1, 0, 2, 3).reshape(T, H_q, D)
  diff_3 = jnp.abs(attn_test3_flat - attn_ref)
  max_diff_3 = float(jnp.max(diff_3))
  print(
      "  Test 3 (KV-Major + Strided DMA + In-Kernel Norm & RoPE) Max Diff:"
      f" {max_diff_3:.6e}"
  )
  assert max_diff_3 < 1e-4, f"Test 3 diff {max_diff_3} exceeded tolerance"

  # 4. Test 3b Configuration: Token Major Q + Strided DMA + In-Kernel Norm & RoPE
  attn_test3b, _ = wrapper.ragged_paged_attention_rope(
      queries=q_raw_token_major,
      keys=k_norm_ref,
      values=v_raw,
      kv_cache=kv_cache,
      kv_lens=kv_lens,
      page_indices=page_indices,
      cu_q_lens=cu_q_lens,
      distribution=distribution,
      rope_theta=theta,
      rope_dim=D,
      rope_input_ordering="split",
      sm_scale=sm_scale,
      kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
      is_kv_group_major=False,
      use_strided_dma=True,
      apply_rmsnorm=True,
      gamma_q=gamma_q,
  )
  diff_3b = jnp.abs(attn_test3b - attn_ref)
  max_diff_3b = float(jnp.max(diff_3b))
  print(
      "  Test 3b (Token Major + Strided DMA + In-Kernel Norm & RoPE) Max Diff:"
      f" {max_diff_3b:.6e}"
  )
  assert max_diff_3b < 1e-4, f"Test 3b diff {max_diff_3b} exceeded tolerance"

  print(">>> TEST 6 PASSED <<<")


def test_three_pipelines_equivalence():
  print("=" * 80)
  print("TEST 7: THREE PREFILL PIPELINES NUMERICAL PARITY (CPU / HOST)")
  print("=" * 80)

  T = 64
  H_in = 256
  H_q = 8
  H_kv = 2
  D = 128
  G = H_q // H_kv

  key = jax.random.PRNGKey(99)
  k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

  x_mid = jax.random.normal(k1, (T, H_in), dtype=jnp.bfloat16)
  w_q = jax.random.normal(k2, (H_in, H_q * D), dtype=jnp.bfloat16) * 0.02
  w_k = jax.random.normal(k3, (H_in, H_kv * D), dtype=jnp.bfloat16) * 0.02
  w_v = jax.random.normal(k4, (H_in, H_kv * D), dtype=jnp.bfloat16) * 0.02

  gamma_in = jnp.ones((H_in,), dtype=jnp.bfloat16)
  gamma_q = jnp.ones((D,), dtype=jnp.bfloat16)
  gamma_k = jnp.ones((D,), dtype=jnp.bfloat16)

  # Weights
  w_q_kv = qkv_pipeline.permute_q_weight_to_kv_group_major(w_q, H_kv, D)
  w_kv_joint = qkv_pipeline.permute_joint_kv_weights(w_k, w_v, H_kv, D)

  w_q_fp8, scale_w_q = qkv_pipeline.quantize_q_kv_major_weight_to_fp8(w_q_kv)
  w_kv_fp8, scale_w_kv = qkv_pipeline.quantize_joint_kv_weight_to_fp8(
      w_kv_joint
  )
  w_q_base_fp8, scale_w_q_base = qkv_pipeline.quantize_weight_to_fp8_static(w_q)
  w_k_fp8, scale_w_k = qkv_pipeline.quantize_weight_to_fp8_static(w_k)
  w_v_fp8, scale_w_v = qkv_pipeline.quantize_weight_to_fp8_static(w_v)

  # Baseline
  q_base, k_base, v_base = qkv_pipeline.qkv_projection_pipeline_baseline(
      x_mid=x_mid,
      w_q_base=w_q_base_fp8,
      w_k_base=w_k_fp8,
      w_v_base=w_v_fp8,
      scale_w_q=scale_w_q_base,
      scale_w_k=scale_w_k,
      scale_w_v=scale_w_v,
      gamma_input=gamma_in,
      gamma_q=gamma_q,
      gamma_k=gamma_k,
      num_kv_heads=H_kv,
      head_dim=D,
      use_in_kernel_rope=False,
  )

  # Test 1 Pipeline: KV head Major Q + joint KV + In-Kernel Norm and RoPE
  q_t1_raw, k_t1, v_t1, _ = (
      qkv_pipeline.qkv_projection_pipeline_head_major_q_joint_kv(
          x_mid=x_mid,
          w_q_kv_major=w_q_fp8,
          w_kv_joint=w_kv_fp8,
          scale_w_q=scale_w_q,
          scale_w_kv=scale_w_kv,
          gamma_input=gamma_in,
          gamma_q=gamma_q,
          gamma_k=gamma_k,
          num_kv_heads=H_kv,
          head_dim=D,
          use_in_kernel_norm=True,
          use_in_kernel_rope=True,
      )
  )
  # Emulate in-kernel norm + RoPE on Test 1 Q
  q_t1_norm = qkv_pipeline.head_rms_norm(q_t1_raw, gamma_q)
  q_t1_rot = qkv_pipeline.apply_reference_rope_5d(q_t1_norm, ordering="split")
  q_t1_flat = q_t1_rot.transpose(1, 0, 2, 3).reshape(T, H_q, D)

  cos_sim_1 = float(
      jnp.sum(q_base.astype(jnp.float32) * q_t1_flat.astype(jnp.float32))
      / (
          jnp.linalg.norm(q_base.astype(jnp.float32))
          * jnp.linalg.norm(q_t1_flat.astype(jnp.float32))
      )
  )
  print(f"  Test 1 Pipeline Q vs Baseline Cosine Similarity: {cos_sim_1:.6f}")
  assert cos_sim_1 > 0.999, f"Test 1 Q similarity {cos_sim_1} too low"

  # Test 2 Pipeline: Token Major Q + separate Q Norm + joint KV + Strided DMA + In-Kernel RoPE
  q_t2_norm, k_t2, v_t2, _ = (
      qkv_pipeline.qkv_projection_pipeline_token_major_joint_kv(
          x_mid=x_mid,
          w_q_base=w_q_base_fp8,
          w_kv_joint=w_kv_joint,
          scale_w_q=scale_w_q_base,
          scale_w_kv=scale_w_kv,
          gamma_input=gamma_in,
          gamma_q=gamma_q,
          gamma_k=gamma_k,
          num_kv_heads=H_kv,
          head_dim=D,
          use_in_kernel_norm=False,
          use_in_kernel_rope=True,
      )
  )
  # Emulate in-kernel RoPE on Test 2 Q
  q_t2_rot = apply_reference_rope(q_t2_norm, ordering="split")
  cos_sim_2 = float(
      jnp.sum(q_base.astype(jnp.float32) * q_t2_rot.astype(jnp.float32))
      / (
          jnp.linalg.norm(q_base.astype(jnp.float32))
          * jnp.linalg.norm(q_t2_rot.astype(jnp.float32))
      )
  )
  print(f"  Test 2 Pipeline Q vs Baseline Cosine Similarity: {cos_sim_2:.6f}")
  assert cos_sim_2 > 0.999, f"Test 2 Q similarity {cos_sim_2} too low"

  # Test 3 Pipeline: Token Major Q (or KV-Major) with In-Kernel Norm and RoPE
  q_t3_raw, k_t3, v_t3, _ = (
      qkv_pipeline.qkv_projection_pipeline_token_major_joint_kv(
          x_mid=x_mid,
          w_q_base=w_q_base_fp8,
          w_kv_joint=w_kv_joint,
          scale_w_q=scale_w_q_base,
          scale_w_kv=scale_w_kv,
          gamma_input=gamma_in,
          gamma_q=gamma_q,
          gamma_k=gamma_k,
          num_kv_heads=H_kv,
          head_dim=D,
          use_in_kernel_norm=True,
          use_in_kernel_rope=True,
      )
  )
  q_t3_norm = qkv_pipeline.head_rms_norm(q_t3_raw, gamma_q)
  q_t3_rot = apply_reference_rope(q_t3_norm, ordering="split")
  cos_sim_3 = float(
      jnp.sum(q_base.astype(jnp.float32) * q_t3_rot.astype(jnp.float32))
      / (
          jnp.linalg.norm(q_base.astype(jnp.float32))
          * jnp.linalg.norm(q_t3_rot.astype(jnp.float32))
      )
  )
  print(f"  Test 3 Pipeline Q vs Baseline Cosine Similarity: {cos_sim_3:.6f}")
  assert cos_sim_3 > 0.999, f"Test 3 Q similarity {cos_sim_3} too low"

  print(">>> TEST 7 PASSED <<<")


def test_fused_pallas_norm_rope_correctness():
  print("=" * 80)
  print("TEST 8: FUSED PALLAS RMSNORM + RoPE NUMERICAL PARITY")
  print("=" * 80)

  H_kv = 4
  T = 128
  G = 8
  D = 128
  theta = 1000000.0

  key = jax.random.PRNGKey(42)
  k1, k2 = jax.random.split(key, 2)
  q_raw = jax.random.normal(k1, (H_kv, T, G, D), dtype=jnp.bfloat16)
  gamma_q = jax.random.normal(k2, (D,), dtype=jnp.bfloat16)

  # Sequential Reference: Head RMSNorm -> Reference RoPE
  q_norm_ref = qkv_pipeline.head_rms_norm(q_raw, gamma_q)
  q_rot_ref = qkv_pipeline.apply_reference_rope_5d(
      q_norm_ref, theta=theta, ordering="split"
  )

  # Fused Pallas RMSNorm + RoPE
  q_rot_fused = pallas_rmsnorm.pallas_2d_rmsnorm_rope(
      q_raw,
      gamma_q,
      theta=theta,
      ordering="split",
  )

  cos_sim = float(
      jnp.sum(q_rot_ref.astype(jnp.float32) * q_rot_fused.astype(jnp.float32))
      / (
          jnp.linalg.norm(q_rot_ref.astype(jnp.float32))
          * jnp.linalg.norm(q_rot_fused.astype(jnp.float32))
      )
  )
  max_diff = float(
      jnp.max(
          jnp.abs(
              q_rot_ref.astype(jnp.float32) - q_rot_fused.astype(jnp.float32)
          )
      )
  )

  print(f"  Fused Norm+RoPE vs Sequential Reference Cosine Sim: {cos_sim:.8f}")
  print(f"  Fused Norm+RoPE vs Sequential Reference Max Diff:   {max_diff:.8e}")
  assert cos_sim > 0.9999, f"Cosine similarity {cos_sim} too low"

  # Also test 3D Token-Major input: [T, H_q, D]
  H_q = H_kv * G
  q_3d = q_raw.reshape(H_kv, T, G, D).transpose((1, 0, 2, 3)).reshape(T, H_q, D)
  q_3d_norm_ref = qkv_pipeline.head_rms_norm(q_3d, gamma_q)
  q_3d_rot_ref = apply_reference_rope(
      q_3d_norm_ref, theta=theta, ordering="split"
  )

  q_3d_rot_fused = pallas_rmsnorm.pallas_2d_rmsnorm_rope(
      q_3d,
      gamma_q,
      theta=theta,
      ordering="split",
  )
  cos_sim_3d = float(
      jnp.sum(
          q_3d_rot_ref.astype(jnp.float32) * q_3d_rot_fused.astype(jnp.float32)
      )
      / (
          jnp.linalg.norm(q_3d_rot_ref.astype(jnp.float32))
          * jnp.linalg.norm(q_3d_rot_fused.astype(jnp.float32))
      )
  )
  print(
      f"  3D Token-Major Fused Norm+RoPE Cosine Sim:          {cos_sim_3d:.8f}"
  )
  assert cos_sim_3d > 0.9999, f"3D Cosine similarity {cos_sim_3d} too low"

  # Test Pipeline Equivalence (Test 5 / Solution C with use_pallas_fused_norm_rope)
  x_mid = jax.random.normal(key, (T, 5120), dtype=jnp.bfloat16)
  w_q = jax.random.normal(k1, (H_kv, 5120, G * D), dtype=jnp.bfloat16)
  w_kv = jax.random.normal(k2, (5120, 2 * H_kv * D), dtype=jnp.bfloat16)
  scale_w = jnp.ones((1,), dtype=jnp.bfloat16)
  gamma_in = jnp.ones((5120,), dtype=jnp.bfloat16)
  gamma_k = jnp.ones((D,), dtype=jnp.bfloat16)

  # Pipeline Reference: Head RMSNorm + Reference RoPE (not in-kernel)
  q_pipe_ref, _, _, _ = (
      qkv_pipeline.qkv_projection_pipeline_head_major_q_joint_kv(
          x_mid=x_mid,
          w_q_kv_major=w_q,
          w_kv_joint=w_kv,
          scale_w_q=scale_w,
          scale_w_kv=scale_w,
          gamma_input=gamma_in,
          gamma_q=gamma_q,
          gamma_k=gamma_k,
          num_kv_heads=H_kv,
          head_dim=D,
          use_in_kernel_norm=False,
          use_in_kernel_rope=False,
      )
  )

  # Pipeline Fused: Dedicated Fused 2D Pallas RMSNorm + RoPE
  q_pipe_fused, _, _, _ = (
      qkv_pipeline.qkv_projection_pipeline_head_major_q_joint_kv(
          x_mid=x_mid,
          w_q_kv_major=w_q,
          w_kv_joint=w_kv,
          scale_w_q=scale_w,
          scale_w_kv=scale_w,
          gamma_input=gamma_in,
          gamma_q=gamma_q,
          gamma_k=gamma_k,
          num_kv_heads=H_kv,
          head_dim=D,
          use_pallas_fused_norm_rope=True,
          use_in_kernel_rope=False,
      )
  )
  cos_sim_pipe = float(
      jnp.sum(
          q_pipe_ref.astype(jnp.float32)
          * q_pipe_fused.reshape(H_kv, T, G, D).astype(jnp.float32)
      )
      / (
          jnp.linalg.norm(q_pipe_ref.astype(jnp.float32))
          * jnp.linalg.norm(q_pipe_fused.astype(jnp.float32))
      )
  )
  print(
      "  Pipeline Fused Norm+RoPE Execution Cosine Sim:     "
      f" {cos_sim_pipe:.8f}"
  )
  assert (
      cos_sim_pipe > 0.9999
  ), f"Pipeline Cosine similarity {cos_sim_pipe} too low"

  print(">>> TEST 8 PASSED <<<")


def test_token_major_out_correctness():
  print("=" * 80)
  print("TEST 9: DIRECT TOKEN-MAJOR ATTENTION OUTPUT (ZERO-TRANSPOSE OUT)")
  print("=" * 80)

  T = 128
  H_q = 8
  H_kv = 2
  G = H_q // H_kv  # 4
  D = 128
  theta = 1000000.0
  sm_scale = 1.0 / np.sqrt(D)
  dtype = jnp.float32

  key = jax.random.PRNGKey(42)
  k1, k2, k3 = jax.random.split(key, 3)

  q_raw_token_major = jax.random.normal(k1, (T, H_q, D), dtype=dtype)
  k_raw = jax.random.normal(k2, (T, H_kv, D), dtype=dtype)
  v_raw = jax.random.normal(k3, (T, H_kv, D), dtype=dtype)

  # Permute Q to KV-Group Major: [H_kv, T, G, D]
  q_kv_major = q_raw_token_major.reshape(T, H_kv, G, D).transpose(1, 0, 2, 3)

  # Reference attention output: [T, H_q, D]
  q_rot_ref = apply_reference_rope(
      q_raw_token_major, theta=theta, ordering="split"
  )
  k_rot_ref = apply_reference_rope(k_raw, theta=theta, ordering="split")
  attn_ref = reference_attention(q_rot_ref, k_rot_ref, v_raw, sm_scale=sm_scale)

  page_size = 16
  num_pages = (T + page_size - 1) // page_size + 4
  kv_cache = jnp.zeros(
      wrapper.get_kv_cache_shape(
          total_num_pages=num_pages,
          page_size=page_size,
          actual_num_kv_heads=H_kv,
          actual_head_dim=D,
          kv_dtype=dtype,
          kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
      ),
      dtype=dtype,
  )
  kv_lens = jnp.array([T], dtype=jnp.int32)
  page_indices = jnp.arange(num_pages, dtype=jnp.int32)
  cu_q_lens = jnp.array([0, T], dtype=jnp.int32)
  distribution = jnp.array([0, 1, 1], dtype=jnp.int32)

  # Run bRPA with Head-Major input Q AND out_token_major=True (dedicated separate HBM output)
  attn_tok_out, _ = wrapper.ragged_paged_attention_rope(
      queries=q_kv_major,
      keys=k_raw,
      values=v_raw,
      kv_cache=kv_cache,
      kv_lens=kv_lens,
      page_indices=page_indices,
      cu_q_lens=cu_q_lens,
      distribution=distribution,
      rope_theta=theta,
      rope_dim=D,
      rope_input_ordering="split",
      sm_scale=sm_scale,
      kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
      is_kv_group_major=True,
      out_token_major=True,
  )

  # Assert output is directly token-major [T, H_q, D]
  assert attn_tok_out.shape == (
      T,
      H_q,
      D,
  ), f"Expected shape {(T, H_q, D)}, got {attn_tok_out.shape}"

  diff = jnp.abs(attn_tok_out - attn_ref)
  max_diff = float(jnp.max(diff))
  mean_diff = float(jnp.mean(diff))
  cos_sim = float(
      jnp.sum(attn_tok_out.astype(jnp.float32) * attn_ref.astype(jnp.float32))
      / (
          jnp.linalg.norm(attn_tok_out.astype(jnp.float32))
          * jnp.linalg.norm(attn_ref.astype(jnp.float32))
      )
  )
  print(
      f"  Token-Major Out vs Ref: Max diff: {max_diff:.6e}, Cosine Sim:"
      f" {cos_sim:.8f}"
  )
  assert max_diff < 1e-4, f"Max diff {max_diff} exceeded tolerance"
  assert cos_sim > 0.9999, f"Cosine similarity {cos_sim} too low"

  # Also verify with token-major input Q (aliased buffer path)
  attn_tok_alias, _ = wrapper.ragged_paged_attention_rope(
      queries=q_raw_token_major,
      keys=k_raw,
      values=v_raw,
      kv_cache=kv_cache,
      kv_lens=kv_lens,
      page_indices=page_indices,
      cu_q_lens=cu_q_lens,
      distribution=distribution,
      rope_theta=theta,
      rope_dim=D,
      rope_input_ordering="split",
      sm_scale=sm_scale,
      kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
      is_kv_group_major=False,
      out_token_major=True,
  )
  assert attn_tok_alias.shape == (T, H_q, D)
  alias_diff = float(jnp.max(jnp.abs(attn_tok_out - attn_tok_alias)))
  print(f"  Dedicated Buffer vs Aliased Buffer Max Diff: {alias_diff:.6e}")
  assert alias_diff < 1e-4, f"Alias diff {alias_diff} too high"

  print(">>> TEST 9 PASSED <<<")


def test_head_sharded_joint_kv_correctness():
  print("=" * 80)
  print("TEST 10: HEAD-SHARDED JOINT W_KV PIPELINE & TP=2 COLUMN PARTITIONING")
  print("=" * 80)

  T = 128
  H_in = 5120
  H_q = 32
  H_kv = 8
  G = H_q // H_kv  # 4
  D = 128

  key = jax.random.PRNGKey(42)
  k1, k2, k3, k4 = jax.random.split(key, 4)

  x_mid = jax.random.normal(k1, (T, H_in), dtype=jnp.bfloat16)
  w_q = jax.random.normal(k2, (H_in, H_q * D), dtype=jnp.bfloat16)
  w_k = jax.random.normal(k3, (H_in, H_kv * D), dtype=jnp.bfloat16)
  w_v = jax.random.normal(k4, (H_in, H_kv * D), dtype=jnp.bfloat16)

  gamma_in = jnp.ones((H_in,), dtype=jnp.bfloat16)
  gamma_q = jnp.ones((D,), dtype=jnp.bfloat16)
  gamma_k = jnp.ones((D,), dtype=jnp.bfloat16)

  # 1. Offline weight permutations (Load-time)
  w_q_kv_major = qkv_pipeline.permute_q_weight_to_kv_group_major(
      w_q, num_kv_heads=H_kv, head_dim=D
  )
  w_kv_prev = qkv_pipeline.permute_joint_kv_weights(
      w_k, w_v, num_kv_heads=H_kv, head_dim=D
  )
  w_kv_head_sharded = qkv_pipeline.permute_joint_kv_weights_head_sharded(
      w_k, w_v, num_kv_heads=H_kv, head_dim=D
  )

  # Quantize weights
  w_q_fp8, scale_w_q = qkv_pipeline.quantize_q_kv_major_weight_to_fp8(
      w_q_kv_major
  )
  w_kv_prev_fp8, scale_w_kv_prev = qkv_pipeline.quantize_joint_kv_weight_to_fp8(
      w_kv_prev
  )
  w_kv_sharded_fp8, scale_w_kv_sharded = (
      qkv_pipeline.quantize_head_sharded_joint_kv_weight_to_fp8(
          w_kv_head_sharded
      )
  )

  # 2. Run previous pipeline
  q_prev, k_prev, v_prev, kv_prev = (
      qkv_pipeline.qkv_projection_pipeline_head_major_q_joint_kv(
          x_mid=x_mid,
          w_q_kv_major=w_q_fp8,
          w_kv_joint=w_kv_prev_fp8,
          scale_w_q=scale_w_q,
          scale_w_kv=scale_w_kv_prev,
          gamma_input=gamma_in,
          gamma_q=gamma_q,
          gamma_k=gamma_k,
          num_kv_heads=H_kv,
          head_dim=D,
          use_in_kernel_norm=False,
          use_in_kernel_rope=False,
      )
  )

  # 3. Run new head-sharded joint KV pipeline
  q_sharded, k_sharded, v_sharded, kv_sharded = (
      qkv_pipeline.qkv_projection_pipeline_head_sharded_joint_kv(
          x_mid=x_mid,
          w_q_kv_major=w_q_fp8,
          w_kv_head_sharded=w_kv_sharded_fp8,
          scale_w_q=scale_w_q,
          scale_w_kv=scale_w_kv_sharded,
          gamma_input=gamma_in,
          gamma_q=gamma_q,
          gamma_k=gamma_k,
          num_kv_heads=H_kv,
          head_dim=D,
          use_in_kernel_norm=False,
          use_in_kernel_rope=False,
      )
  )

  # Assert full numerical equivalence
  q_cos = float(
      jnp.sum(q_prev.astype(jnp.float32) * q_sharded.astype(jnp.float32))
      / (
          jnp.linalg.norm(q_prev.astype(jnp.float32))
          * jnp.linalg.norm(q_sharded.astype(jnp.float32))
      )
  )
  k_diff = float(
      jnp.max(
          jnp.abs(k_prev.astype(jnp.float32) - k_sharded.astype(jnp.float32))
      )
  )
  v_diff = float(
      jnp.max(
          jnp.abs(v_prev.astype(jnp.float32) - v_sharded.astype(jnp.float32))
      )
  )
  kv_diff = float(
      jnp.max(
          jnp.abs(kv_prev.astype(jnp.float32) - kv_sharded.astype(jnp.float32))
      )
  )

  print(f"  Head-Sharded Q Cosine Sim:                 {q_cos:.8f}")
  print(f"  Head-Sharded K Max Absolute Difference:     {k_diff:.6e}")
  print(f"  Head-Sharded V Max Absolute Difference:     {v_diff:.6e}")
  print(f"  Head-Sharded Joint KV Max Absolute Diff:    {kv_diff:.6e}")

  assert q_cos > 0.9999, f"Q Cosine similarity {q_cos} too low"
  assert k_diff == 0.0, f"K difference {k_diff} is non-zero"
  assert v_diff == 0.0, f"V difference {v_diff} is non-zero"
  assert kv_diff == 0.0, f"Joint KV difference {kv_diff} is non-zero"

  # 4. Simulate TP=2 Column Partitioning (Verifying zero cross-device communication)
  total_cols = H_kv * 2 * D  # 8 * 2 * 128 = 2048
  cols_per_dev = total_cols // 2  # 1024
  h_kv_per_dev = H_kv // 2  # 4

  x_norm = qkv_pipeline.head_rms_norm(x_mid, gamma_in)
  x_fp8, scale_x = qkv_pipeline.quantize_to_fp8_dynamic(x_norm)

  # Device 0: columns [0 .. 1024]
  w_dev0 = w_kv_sharded_fp8[:, :cols_per_dev]
  scale_dev0 = scale_w_kv_sharded[:, :cols_per_dev]
  gemm_dev0 = (
      jnp.dot(x_fp8.astype(jnp.float32), w_dev0.astype(jnp.float32))
      * scale_x.astype(jnp.float32)
      * scale_dev0.astype(jnp.float32)
  ).astype(jnp.bfloat16)
  gemm_dev0_4d = gemm_dev0.reshape(T, h_kv_per_dev, 2, D)
  k_dev0 = gemm_dev0_4d[:, :, 0, :]  # Heads 0..3 of K
  v_dev0 = gemm_dev0_4d[:, :, 1, :]  # Heads 0..3 of V

  # Device 1: columns [1024 .. 2048]
  w_dev1 = w_kv_sharded_fp8[:, cols_per_dev:]
  scale_dev1 = scale_w_kv_sharded[:, cols_per_dev:]
  gemm_dev1 = (
      jnp.dot(x_fp8.astype(jnp.float32), w_dev1.astype(jnp.float32))
      * scale_x.astype(jnp.float32)
      * scale_dev1.astype(jnp.float32)
  ).astype(jnp.bfloat16)
  gemm_dev1_4d = gemm_dev1.reshape(T, h_kv_per_dev, 2, D)
  k_dev1 = gemm_dev1_4d[:, :, 0, :]  # Heads 4..7 of K
  v_dev1 = gemm_dev1_4d[:, :, 1, :]  # Heads 4..7 of V

  # Reassemble across device partitions
  k_reassembled = jnp.concatenate([k_dev0, k_dev1], axis=1)
  v_reassembled = jnp.concatenate([v_dev0, v_dev1], axis=1)

  # Compare against full-rank GEMM output
  k_raw_full = (
      (
          jnp.dot(
              x_fp8.astype(jnp.float32), w_kv_sharded_fp8.astype(jnp.float32)
          )
          * scale_x.astype(jnp.float32)
          * scale_w_kv_sharded.astype(jnp.float32)
      )
      .astype(jnp.bfloat16)
      .reshape(T, H_kv, 2, D)[:, :, 0, :]
  )

  v_raw_full = (
      (
          jnp.dot(
              x_fp8.astype(jnp.float32), w_kv_sharded_fp8.astype(jnp.float32)
          )
          * scale_x.astype(jnp.float32)
          * scale_w_kv_sharded.astype(jnp.float32)
      )
      .astype(jnp.bfloat16)
      .reshape(T, H_kv, 2, D)[:, :, 1, :]
  )

  tp_k_diff = float(
      jnp.max(
          jnp.abs(
              k_reassembled.astype(jnp.float32) - k_raw_full.astype(jnp.float32)
          )
      )
  )
  tp_v_diff = float(
      jnp.max(
          jnp.abs(
              v_reassembled.astype(jnp.float32) - v_raw_full.astype(jnp.float32)
          )
      )
  )

  print(f"  TP=2 Partitioned K Assembly Max Diff:       {tp_k_diff:.6e}")
  print(f"  TP=2 Partitioned V Assembly Max Diff:       {tp_v_diff:.6e}")
  assert tp_k_diff == 0.0, f"TP=2 K assembly diff {tp_k_diff} is non-zero"
  assert tp_v_diff == 0.0, f"TP=2 V assembly diff {tp_v_diff} is non-zero"

  print(">>> TEST 10 PASSED <<<")


def test_head_major_q_separate_kv_correctness():
  """Verifies numerical equivalence of Head-Major Q + Separate W_K, W_V Pipeline."""
  print("\n==================================================")
  print("TEST 11: Head-Major Q + Separate W_K, W_V Pipeline Correctness")
  print("==================================================")
  key = jax.random.PRNGKey(42)
  T, H_in = 128, 5120
  H_q, H_kv, D = 32, 4, 128
  gamma_in = jnp.ones((H_in,), dtype=jnp.bfloat16)
  gamma_q = jnp.ones((D,), dtype=jnp.bfloat16)
  gamma_k = jnp.ones((D,), dtype=jnp.bfloat16)

  k_x, k_wq, k_wk, k_wv = jax.random.split(key, 4)
  x_mid = jax.random.normal(k_x, (T, H_in), dtype=jnp.bfloat16)
  w_q = jax.random.normal(k_wq, (H_in, H_q * D), dtype=jnp.bfloat16) * 0.02
  w_k = jax.random.normal(k_wk, (H_in, H_kv * D), dtype=jnp.bfloat16) * 0.02
  w_v = jax.random.normal(k_wv, (H_in, H_kv * D), dtype=jnp.bfloat16) * 0.02

  w_q_base_fp8, scale_w_q_base = qkv_pipeline.quantize_weight_to_fp8_static(w_q)
  w_k_fp8, scale_w_k = qkv_pipeline.quantize_weight_to_fp8_static(w_k)
  w_v_fp8, scale_w_v = qkv_pipeline.quantize_weight_to_fp8_static(w_v)

  w_q_kv_major = qkv_pipeline.permute_q_weight_to_kv_group_major(w_q, H_kv, D)
  w_q_fp8, scale_w_q = qkv_pipeline.quantize_q_kv_major_weight_to_fp8(
      w_q_kv_major
  )

  # Baseline Separate Pipeline
  q_base, k_base, v_base = qkv_pipeline.qkv_projection_pipeline_baseline(
      x_mid=x_mid,
      w_q_base=w_q_base_fp8,
      w_k_base=w_k_fp8,
      w_v_base=w_v_fp8,
      scale_w_q=scale_w_q_base,
      scale_w_k=scale_w_k,
      scale_w_v=scale_w_v,
      gamma_input=gamma_in,
      gamma_q=gamma_q,
      gamma_k=gamma_k,
      num_kv_heads=H_kv,
      head_dim=D,
      use_in_kernel_rope=False,
  )

  # Head-Major Q + Separate W_K, W_V Pipeline
  q_sep, k_sep, v_sep, kv_sep = (
      qkv_pipeline.qkv_projection_pipeline_head_major_q_separate_kv(
          x_mid=x_mid,
          w_q_kv_major=w_q_fp8,
          w_k_base=w_k_fp8,
          w_v_base=w_v_fp8,
          scale_w_q=scale_w_q,
          scale_w_k=scale_w_k,
          scale_w_v=scale_w_v,
          gamma_input=gamma_in,
          gamma_q=gamma_q,
          gamma_k=gamma_k,
          num_kv_heads=H_kv,
          head_dim=D,
          use_in_kernel_rope=False,
      )
  )

  # Verify Q equivalence (transpose from [H_kv, T, G, D] -> [T, H_q, D])
  q_sep_flat = q_sep.transpose(1, 0, 2, 3).reshape(T, H_q, D)
  cos_sim_q = float(
      jnp.sum(q_base.astype(jnp.float32) * q_sep_flat.astype(jnp.float32))
      / (
          jnp.linalg.norm(q_base.astype(jnp.float32))
          * jnp.linalg.norm(q_sep_flat.astype(jnp.float32))
      )
  )
  print(f"  Head-Major Q Cosine Sim:               {cos_sim_q:.8f}")
  assert cos_sim_q > 0.9999, f"Q cosine similarity {cos_sim_q} too low"

  # Verify K bit-exact equivalence
  diff_k = float(
      jnp.max(
          jnp.abs(
              k_base.view(jnp.int8).astype(jnp.float32)
              - k_sep.view(jnp.int8).astype(jnp.float32)
          )
      )
  )
  print(f"  Separate K Max Absolute Difference:    {diff_k:.6e}")
  assert diff_k == 0.0, f"K max difference {diff_k} is non-zero"

  # Verify V bit-exact equivalence
  diff_v = float(
      jnp.max(
          jnp.abs(
              v_base.view(jnp.int8).astype(jnp.float32)
              - v_sep.view(jnp.int8).astype(jnp.float32)
          )
      )
  )
  print(f"  Separate V Max Absolute Difference:    {diff_v:.6e}")
  assert diff_v == 0.0, f"V max difference {diff_v} is non-zero"

  # Verify joint KV packing format
  assert kv_sep.shape == (T, 2, H_kv, D)
  print("  KV Cache Joint Format Shape:           Verified [T, 2, H_kv, D]")
  print(">>> TEST 11 PASSED <<<")


def main(_):
  has_tpu = any("tpu" in d.device_kind.lower() for d in jax.devices())

  if has_tpu:
    test_in_kernel_rope_correctness()
    test_kv_group_major_correctness()
    test_strided_dma_correctness()
    test_in_kernel_norm_rope_tpu_correctness()
    test_token_major_out_correctness()
  else:
    print("[NOTE] Skipping Pallas TPU kernel tests on CPU host.")

  test_qkv_pipeline_equivalence()
  test_head_major_q_joint_kv_correctness()
  test_three_pipelines_equivalence()
  test_fused_pallas_norm_rope_correctness()
  test_head_sharded_joint_kv_correctness()
  test_head_major_q_separate_kv_correctness()
  print("\n==================================================")
  print("ALL NUMERICAL AND PIPELINE TESTS PASSED")
  print("==================================================")


if __name__ == "__main__":
  app.run(main)
