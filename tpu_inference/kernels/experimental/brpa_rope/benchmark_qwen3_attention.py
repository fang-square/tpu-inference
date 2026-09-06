# Copyright 2026 Google LLC
#
# Microbenchmark suite for Qwen3-32B attention on TPU:
# Comparing Baseline (Standalone RoPE + Copies) vs Fused brpa_rope (In-Kernel RoPE).

import math
import os
import time

if "--xla_tpu_use_dynamic_smem_negotiation" not in os.environ.get(
    "LIBTPU_INIT_ARGS", ""
):
  os.environ["LIBTPU_INIT_ARGS"] = (
      "--xla_tpu_use_dynamic_smem_negotiation=true "
      + os.environ.get("LIBTPU_INIT_ARGS", "")
  )

from absl import app
from absl import flags
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

try:
  from google3.perftools.accelerators.xprof.api.python import xprof_analysis_client
  from google3.perftools.accelerators.xprof.api.python import xprof_session
  HAS_XPROF = True
except ImportError:
  HAS_XPROF = False

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_reqs", 4, "Number of concurrent prefill requests.")
flags.DEFINE_integer("seq_len", 1024, "Sequence length per prefill request.")
flags.DEFINE_integer("warmup_steps", 5, "Number of warmup iterations.")
flags.DEFINE_integer("benchmark_steps", 20, "Number of timed benchmark iterations.")
flags.DEFINE_integer("page_size", 256, "Page size for paged KV cache.")
flags.DEFINE_integer("bq_sz", 512, "Prefill block query size.")
flags.DEFINE_integer("bkv_sz", 256, "Prefill block KV size.")
flags.DEFINE_integer("bq_c_sz", 512, "Prefill block query compute chunk size.")


def apply_reference_rope_jax(
    x: jax.Array, positions: jax.Array, theta: float = 1000000.0
) -> jax.Array:
  """JAX implementation of standalone RoPE."""
  T, H, D = x.shape
  half_D = D // 2
  inv_freq = 1.0 / (theta ** (jnp.arange(0, D, 2, dtype=jnp.float32) / D))
  freqs = jnp.outer(positions.astype(jnp.float32), inv_freq)
  cos = jnp.cos(freqs)[:, None, :]
  sin = jnp.sin(freqs)[:, None, :]

  x_f32 = x.astype(jnp.float32)
  x1 = x_f32[..., :half_D]
  x2 = x_f32[..., half_D:]
  rot1 = x1 * cos - x2 * sin
  rot2 = x2 * cos + x1 * sin
  return jnp.concatenate([rot1, rot2], axis=-1).astype(x.dtype)


def benchmark_qwen3_attention():
  print("=" * 80)
  print("BENCHMARK: Qwen3-32B Attention (Baseline Standalone RoPE vs Fused In-Kernel RoPE)")
  print("=" * 80)

  num_reqs = FLAGS.num_reqs
  seq_len = FLAGS.seq_len
  total_tokens = num_reqs * seq_len
  page_size = FLAGS.page_size

  # Qwen3-32B per-chip shard (TP=2)
  H_q = 32
  H_kv = 4
  D = 128
  theta = 1000000.0
  dtype = jnp.bfloat16

  print(f"Configuration: {num_reqs} requests x {seq_len} tokens = {total_tokens} total tokens")
  print(f"Per-chip shard: H_q={H_q}, H_kv={H_kv}, D={D} (8:1 GQA)")
  print(f"Page size: {page_size}, Tiling: bq_sz={FLAGS.bq_sz}, bkv_sz={FLAGS.bkv_sz}, bq_c_sz={FLAGS.bq_c_sz}")
  print(f"Data type: {dtype}")

  cu_q_lens = jnp.arange(0, (num_reqs + 1) * seq_len, seq_len, dtype=jnp.int32)
  kv_lens = jnp.full((num_reqs,), seq_len, dtype=jnp.int32)
  distribution = jnp.array([0, 0, num_reqs], dtype=jnp.int32)
  positions = jnp.tile(jnp.arange(seq_len, dtype=jnp.int32), num_reqs)

  pages_per_seq = (seq_len + page_size - 1) // page_size + 4
  total_pages = pages_per_seq * num_reqs

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
      bq_sz=FLAGS.bq_sz,
      bkv_sz=FLAGS.bkv_sz,
      bq_c_sz=FLAGS.bq_c_sz,
      batch_size=1,
      n_buffer=3,
  )
  vmem_limit_bytes = 134217728

  key = jax.random.PRNGKey(42)
  k1, k2, k3 = jax.random.split(key, 3)

  q_raw = jax.random.normal(k1, (total_tokens, H_q, D), dtype=dtype)
  k_raw = jax.random.normal(k2, (total_tokens, H_kv, D), dtype=dtype)
  v_raw = jax.random.normal(k3, (total_tokens, H_kv, D), dtype=dtype)

  # --- Baseline JIT function ---
  @jax.jit
  def baseline_step(q, k, v, cache):
    q_rot = apply_reference_rope_jax(q, positions, theta=theta)
    k_rot = apply_reference_rope_jax(k, positions, theta=theta)
    out, new_cache = wrapper.ragged_paged_attention(
        queries=q_rot,
        keys=k_rot,
        values=v,
        kv_cache=cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        sm_scale=1.0 / math.sqrt(D),
        out_dtype=dtype,
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
        decode_block_sizes=decode_blocks,
        prefill_block_sizes=prefill_blocks,
        vmem_limit_bytes=vmem_limit_bytes,
        apply_rope=False,
    )
    return out, new_cache

  # --- Fused brpa_rope JIT function ---
  @jax.jit
  def fused_step(q, k, v, cache):
    out, new_cache = wrapper.ragged_paged_attention_rope(
        queries=q,
        keys=k,
        values=v,
        kv_cache=cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        rope_theta=theta,
        rope_dim=D,
        rope_input_ordering="split",
        sm_scale=1.0 / math.sqrt(D),
        out_dtype=dtype,
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
        decode_block_sizes=decode_blocks,
        prefill_block_sizes=prefill_blocks,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    return out, new_cache

  print("\n--- Warming up kernels ---")
  for _ in range(FLAGS.warmup_steps):
    out_b, _ = baseline_step(q_raw, k_raw, v_raw, kv_cache)
    jax.block_until_ready(out_b)
    out_f, _ = fused_step(q_raw, k_raw, v_raw, kv_cache)
    jax.block_until_ready(out_f)
  print("Warmup complete.")

  # Benchmark Baseline
  print("\n--- Benchmarking Baseline (Standalone RoPE + bRPA) ---")
  latencies_baseline = []
  for _ in range(FLAGS.benchmark_steps):
    t0 = time.perf_counter()
    out, _ = baseline_step(q_raw, k_raw, v_raw, kv_cache)
    jax.block_until_ready(out)
    t1 = time.perf_counter()
    latencies_baseline.append((t1 - t0) * 1e6)  # microseconds

  med_b = float(np.median(latencies_baseline))
  p99_b = float(np.percentile(latencies_baseline, 99))
  print(f"Baseline Latency: Median = {med_b:.2f} µs, P99 = {p99_b:.2f} µs")

  # Benchmark Fused
  print("\n--- Benchmarking Fused brpa_rope (In-Kernel RoPE) ---")
  latencies_fused = []
  for _ in range(FLAGS.benchmark_steps):
    t0 = time.perf_counter()
    out, _ = fused_step(q_raw, k_raw, v_raw, kv_cache)
    jax.block_until_ready(out)
    t1 = time.perf_counter()
    latencies_fused.append((t1 - t0) * 1e6)  # microseconds

  med_f = float(np.median(latencies_fused))
  p99_f = float(np.percentile(latencies_fused, 99))
  print(f"Fused Latency   : Median = {med_f:.2f} µs, P99 = {p99_f:.2f} µs")

  delta = med_b - med_f
  pct = (delta / med_b) * 100.0
  print(f"\nNet Latency Reduction : -{delta:.2f} µs / layer (-{pct:.1f}%)")
  print(f"Extrapolated (64 Layers): -{delta * 64 / 1000.0:.2f} ms total speedup")


def main(_):
  benchmark_qwen3_attention()


if __name__ == "__main__":
  app.run(main)
