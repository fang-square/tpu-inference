"""Numerical parity test for bRPA with In-Kernel RoPE."""

import os
import sys
from absl import app
from absl import logging
import jax
import jax.numpy as jnp
import numpy as np

try:
  from google3.experimental.users.fangfangz.kernels.brpa_rope import configs, qkv_pipeline, utils, wrapper
except (ModuleNotFoundError, ImportError):
  try:
    from experimental.users.fangfangz.kernels.brpa_rope import configs, qkv_pipeline, utils, wrapper
  except (ModuleNotFoundError, ImportError):
    from tpu_inference.kernels.experimental.brpa_rope import configs, qkv_pipeline, utils, wrapper


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


def main(_):
  has_tpu = any("tpu" in d.device_kind.lower() for d in jax.devices())

  if has_tpu:
    test_in_kernel_rope_correctness()
    test_kv_group_major_correctness()
    test_strided_dma_correctness()
    test_in_kernel_norm_rope_tpu_correctness()
  else:
    print("[NOTE] Skipping Pallas TPU kernel tests on CPU host.")

  test_qkv_pipeline_equivalence()
  test_head_major_q_joint_kv_correctness()
  test_three_pipelines_equivalence()
  print("\n==================================================")
  print("ALL NUMERICAL AND PIPELINE TESTS PASSED")
  print("==================================================")


if __name__ == "__main__":
  app.run(main)
