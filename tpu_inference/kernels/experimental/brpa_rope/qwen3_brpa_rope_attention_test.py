# Copyright 2026 Google LLC
#
# Numerical parity and end-to-end integration test suite for
# Qwen3-32B attention with in-kernel RoPE.

import math
from unittest import mock
from absl import app
from absl import logging
import jax
import jax.numpy as jnp
import numpy as np

try:
  from tpu_inference.kernels.experimental.brpa_rope import configs, utils, wrapper
except ModuleNotFoundError:
  try:
    from google3.experimental.users.fangfangz.kernels.brpa_rope import configs, utils, wrapper
  except ModuleNotFoundError:
    from experimental.users.fangfangz.kernels.brpa_rope import configs, utils, wrapper


class MockTpuInfo:
  num_lanes = 128
  num_sublanes = 8
  generation = 6
  vmem_capacity_bytes = 64 * 1024 * 1024
  smem_capacity_bytes = 16 * 1024 * 1024
  mxu_column_size = 128



def apply_reference_rope_np(
    x: np.ndarray,
    positions: np.ndarray,
    theta: float = 1000000.0,
    ordering: str = "split",
) -> np.ndarray:
  """Exact reference NumPy implementation of RoPE."""
  # x shape: [T, H, D] or [B, T, H, D]
  T, H, D = x.shape[-3], x.shape[-2], x.shape[-1]
  half_D = D // 2
  inv_freq = 1.0 / (theta ** (np.arange(0, D, 2, dtype=np.float32) / D))

  # positions: [T] or [B, T]
  freqs = np.outer(positions.reshape(-1), inv_freq).reshape(
      *positions.shape, half_D
  )
  cos = np.cos(freqs)  # [..., half_D]
  sin = np.sin(freqs)  # [..., half_D]

  if x.ndim == 3:
    cos = cos[:, None, :]  # [T, 1, half_D]
    sin = sin[:, None, :]  # [T, 1, half_D]
  elif x.ndim == 4:
    cos = cos[:, :, None, :]  # [B, T, 1, half_D]
    sin = sin[:, :, None, :]  # [B, T, 1, half_D]

  x_f32 = x.astype(np.float32)
  if ordering == "split":
    x1 = x_f32[..., :half_D]
    x2 = x_f32[..., half_D:]
    rot1 = x1 * cos - x2 * sin
    rot2 = x2 * cos + x1 * sin
    out = np.concatenate([rot1, rot2], axis=-1)
  else:
    orig_shape = x_f32.shape
    x_pairs = x_f32.reshape(*orig_shape[:-1], half_D, 2)
    rot1 = x_pairs[..., 0] * cos - x_pairs[..., 1] * sin
    rot2 = x_pairs[..., 1] * cos + x_pairs[..., 0] * sin
    out = np.stack([rot1, rot2], axis=-1).reshape(orig_shape)

  return out.astype(x.dtype)


def reference_ragged_attention_np(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    cu_q_lens: np.ndarray,
    kv_lens: np.ndarray,
    sm_scale: float,
) -> np.ndarray:
  """Exact reference ragged multi-request attention in float64."""
  num_reqs = len(kv_lens)
  out_list = []

  for r in range(num_reqs):
    q_start = cu_q_lens[r]
    q_end = cu_q_lens[r + 1]
    q_len = q_end - q_start
    kv_len = kv_lens[r]

    if q_len == 0:
      continue

    # Slice for this request: [q_len, H_q, D]
    q_r = q[q_start:q_end].astype(np.float64)
    # Cached + new keys/values for this request: [kv_len, H_kv, D]
    k_r = k[:kv_len, r, :, :] if k.ndim == 4 else k[q_start:q_end].astype(np.float64)
    v_r = v[:kv_len, r, :, :] if v.ndim == 4 else v[q_start:q_end].astype(np.float64)

    H_q = q_r.shape[1]
    H_kv = k_r.shape[1]
    G = H_q // H_kv
    D = q_r.shape[2]

    # Repeat KV heads for GQA: [kv_len, H_q, D]
    k_r_rep = np.repeat(k_r, G, axis=1)
    v_r_rep = np.repeat(v_r, G, axis=1)

    # Transpose to [H_q, q_len, D] and [H_q, kv_len, D]
    q_t = np.transpose(q_r, (1, 0, 2))
    k_t = np.transpose(k_r_rep, (1, 0, 2))
    v_t = np.transpose(v_r_rep, (1, 0, 2))

    # Dot product: [H_q, q_len, kv_len]
    scores = np.matmul(q_t, np.transpose(k_t, (0, 2, 1))) * sm_scale

    # Causal mask for the active query window
    # query position i attends to key positions j <= (kv_len - q_len + i)
    mask = np.zeros((q_len, kv_len), dtype=bool)
    context_offset = kv_len - q_len
    for qi in range(q_len):
      max_kj = context_offset + qi
      mask[qi, : max_kj + 1] = True

    scores = np.where(mask[None, :, :], scores, -1e9)

    # Numerically stable softmax
    max_score = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_score)
    probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Weighted sum: [H_q, q_len, D]
    out_r = np.matmul(probs, v_t)
    # Back to [q_len, H_q, D]
    out_list.append(np.transpose(out_r, (1, 0, 2)))

  return np.concatenate(out_list, axis=0)


def run_qwen3_prefill_parity_test():
  """Tests Qwen3-32B TP=2 prefill workload numerical parity."""
  print("=" * 80)
  print("QWEN3-32B PREFILL NUMERICAL PARITY TEST (TP=2 Shard)")
  print("=" * 80)

  # Qwen3-32B per-chip parameters on TP=2:
  # Global: H_q=64, H_kv=8, D=128 (8:1 GQA)
  # Per-chip shard: H_q=32, H_kv=4, D=128
  H_q = 32
  H_kv = 4
  D = 128
  theta = 1000000.0
  sm_scale = 1.0 / math.sqrt(D)
  dtype = jnp.bfloat16

  # Multi-request prefill batch: 2 requests of length 256 each -> Total T=512
  req_lens = [256, 256]
  total_tokens = sum(req_lens)
  num_reqs = len(req_lens)

  cu_q_lens = np.array([0, 256, 512], dtype=np.int32)
  kv_lens = np.array([256, 256], dtype=np.int32)
  distribution = np.array([0, num_reqs, num_reqs], dtype=np.int32)

  # Generate positions for each request
  positions = np.concatenate([np.arange(l, dtype=np.int32) for l in req_lens])

  key = jax.random.PRNGKey(1337)
  k1, k2, k3 = jax.random.split(key, 3)

  q_raw_jax = jax.random.normal(k1, (total_tokens, H_q, D), dtype=jnp.float32)
  k_raw_jax = jax.random.normal(k2, (total_tokens, H_kv, D), dtype=jnp.float32)
  v_raw_jax = jax.random.normal(k3, (total_tokens, H_kv, D), dtype=jnp.float32)

  q_raw_np = np.array(q_raw_jax)
  k_raw_np = np.array(k_raw_jax)
  v_raw_np = np.array(v_raw_jax)

  print(f"Workload: {num_reqs} prefill requests, Total tokens = {total_tokens}")
  print(f"Per-chip heads: H_q={H_q}, H_kv={H_kv}, Head dim D={D} (GQA 8:1)")
  print(f"RoPE theta = {theta}, ordering = split")

  # 1. Reference computation (Full float64 precision)
  q_rot_ref = apply_reference_rope_np(q_raw_np, positions, theta=theta, ordering="split")
  k_rot_ref = apply_reference_rope_np(k_raw_np, positions, theta=theta, ordering="split")
  attn_ref = reference_ragged_attention_np(
      q_rot_ref, k_rot_ref, v_raw_np, cu_q_lens, kv_lens, sm_scale=sm_scale
  )
  print("Reference attention computed successfully.")

  # 2. bRPA In-Kernel RoPE
  page_size = 256
  max_kv_len = max(kv_lens)
  pages_per_req = (max_kv_len + page_size - 1) // page_size + 4
  total_pages = pages_per_req * num_reqs

  kv_cache = jnp.zeros(
      wrapper.get_kv_cache_shape(
          total_num_pages=total_pages,
          page_size=page_size,
          actual_num_kv_heads=H_kv,
          actual_head_dim=D,
          kv_dtype=dtype,
          kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
      ),
      dtype=dtype,
  )
  page_indices = jnp.arange(total_pages, dtype=jnp.int32)

  decode_blocks = configs.BlockSizes(
      bq_sz=1,
      bkv_sz=1024,
      bq_c_sz=1,
      batch_size=8,
      n_buffer=2,
  )
  prefill_blocks = configs.BlockSizes(
      bq_sz=512,
      bkv_sz=256,
      bq_c_sz=512,
      batch_size=1,
      n_buffer=3,
  )
  vmem_limit_bytes = 134217728

  try:
    attn_fused, _ = wrapper.ragged_paged_attention_rope(
        queries=q_raw_jax.astype(dtype),
        keys=k_raw_jax.astype(dtype),
        values=v_raw_jax.astype(dtype),
        kv_cache=kv_cache,
        kv_lens=jnp.array(kv_lens),
        page_indices=page_indices,
        cu_q_lens=jnp.array(cu_q_lens),
        distribution=jnp.array([0, 0, num_reqs], dtype=jnp.int32),
        rope_theta=theta,
        rope_dim=D,
        rope_input_ordering="split",
        sm_scale=sm_scale,
        out_dtype=dtype,
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
        decode_block_sizes=decode_blocks,
        prefill_block_sizes=prefill_blocks,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    print("Fused brpa_rope attention executed successfully!")

    # Compare fused vs reference
    fused_np = np.array(attn_fused).astype(np.float64)
    diff = np.abs(fused_np - attn_ref)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))

    dot_prod = np.sum(fused_np * attn_ref)
    norm_fused = np.linalg.norm(fused_np)
    norm_ref = np.linalg.norm(attn_ref)
    cosine_sim = float(dot_prod / (norm_fused * norm_ref))

    print(f"  Max Absolute Difference : {max_diff:.6e}")
    print(f"  Mean Absolute Difference: {mean_diff:.6e}")
    print(f"  Cosine Similarity       : {cosine_sim:.8f}")

    assert cosine_sim > 0.999, f"Cosine similarity {cosine_sim} below threshold 0.999"
    print(">>> SUCCESS: Qwen3-32B Prefill Parity Verified! <<<")

  except Exception as e:
    print(f"Execution failed: {e}")
    raise e


def main(_):
  if jax.devices()[0].platform != "tpu":
    with mock.patch("jax.experimental.pallas.tpu.get_tpu_info", return_value=MockTpuInfo()):
      run_qwen3_prefill_parity_test()
  else:
    run_qwen3_prefill_parity_test()


if __name__ == "__main__":
  app.run(main)
