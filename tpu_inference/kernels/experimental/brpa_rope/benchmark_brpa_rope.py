"""Microbenchmark for bRPA with In-Kernel RoPE vs Baseline Pipeline on Sunfish TPU."""

import json
import os
import sys
import time

from absl import app
from absl import flags
from absl import logging
import numpy as np

if "--xla_tpu_use_dynamic_smem_negotiation" not in os.environ.get(
    "LIBTPU_INIT_ARGS", ""
):
  os.environ["LIBTPU_INIT_ARGS"] = (
      "--xla_tpu_use_dynamic_smem_negotiation=true "
      + os.environ.get("LIBTPU_INIT_ARGS", "")
  )

# pylint: disable=g-import-not-at-top
import jax
import jax.numpy as jnp

from google3.experimental.users.fangfangz.kernels.brpa_rope import configs
from google3.experimental.users.fangfangz.kernels.brpa_rope import wrapper

try:
  from google3.perftools.accelerators.xprof.api.python import xprof_analysis_client
  from google3.perftools.accelerators.xprof.api.python import xprof_session
except ImportError:
  xprof_analysis_client = None
  xprof_session = None
# pylint: enable=g-import-not-at-top

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_requests", 4, "Number of active concurrent requests.")
flags.DEFINE_integer("q_len", 1024, "Query length per request.")
flags.DEFINE_integer("kv_len", 1024, "KV cache length per request.")
flags.DEFINE_integer("max_num_seqs", 256, "Maximum sequence capacity.")
flags.DEFINE_integer("num_q_heads", 32, "Number of Q heads (TP=2 shard of 64).")
flags.DEFINE_integer("num_kv_heads", 4, "Number of KV heads (TP=2 shard of 8).")
flags.DEFINE_integer("head_dim", 128, "Head dimension (D).")
flags.DEFINE_integer("page_size", 256, "Page size for paged KV cache.")
flags.DEFINE_integer("num_pages", 8192, "Total KV cache pages allocated.")
flags.DEFINE_integer("num_warmup", 5, "Number of warmup iterations.")
flags.DEFINE_integer("num_iters", 20, "Number of timed benchmark iterations.")
flags.DEFINE_integer("bq_sz", 512, "Prefill block query size.")
flags.DEFINE_integer("bkv_sz", 256, "Prefill block KV size.")
flags.DEFINE_integer("bq_c_sz", 512, "Prefill block query compute chunk size.")
flags.DEFINE_float("rope_theta", 1000000.0, "RoPE theta frequency base.")
flags.DEFINE_string(
    "rope_ordering", "split", "RoPE ordering ('split' or 'interleaved')."
)
flags.DEFINE_enum(
    "benchmark_mode",
    "all",
    ["all", "baseline", "fused", "no_rope", "strided_dma", "kv_group_major"],
    "Benchmark mode to run.",
)
flags.DEFINE_bool("record_xprof", True, "Capture programmatic XProf trace.")


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


def get_kernel_durations(session_id):
  """Extract RPAm kernel execution times from the uploaded XProf trace."""
  if not xprof_analysis_client:
    return {"RPAm": []}
  print("Requesting trace analysis from XProf server...", flush=True)
  client = xprof_analysis_client.XprofAnalysisClient()
  jtrace = None
  for attempt in range(10):
    try:
      trace = client.get_profile_data(
          "trace_viewer.json", session_id, rpc_deadline_s=300
      )
      if trace and trace[1]:
        jtrace = json.loads(trace[1])
        break
    except Exception:  # pylint: disable=broad-exception-caught
      pass
    print(
        f"Waiting for trace analysis backend... (attempt {attempt + 1}/10)",
        flush=True,
    )
    time.sleep(5)

  if not jtrace:
    print("Warning: Could not retrieve or parse trace_viewer.json.", flush=True)
    return {"RPAm": []}

  results = {"RPAm": []}
  for e in jtrace.get("traceEvents", []):
    if "name" not in e or "dur" not in e:
      continue
    if "RPAm-p" in e["name"]:
      results["RPAm"].append(e["dur"])
  return results


def run_benchmark():
  """Runs the 5-way microbenchmark on hardware."""
  devices = jax.devices()
  print(f"JAX Devices: {devices}")

  num_requests = FLAGS.num_requests
  max_num_seqs = FLAGS.max_num_seqs
  q_len = FLAGS.q_len
  kv_len = FLAGS.kv_len
  total_tokens = num_requests * q_len
  num_pages = FLAGS.num_pages
  page_size = FLAGS.page_size
  num_q_heads = FLAGS.num_q_heads
  num_kv_heads = FLAGS.num_kv_heads
  head_dim = FLAGS.head_dim
  distribution = jnp.array([0, 0, num_requests], dtype=jnp.int32)
  sm_scale = 1.0 / (head_dim**0.5)
  theta = FLAGS.rope_theta
  ordering = FLAGS.rope_ordering

  print("=" * 85)
  print("BENCHMARKING bRPA: BASELINE vs IN-KERNEL RoPE (Qwen3-32B Prefill)")
  print("=" * 85)
  print("Workload Config:")
  print(f"  Requests: {num_requests}, q_len: {q_len}, kv_len: {kv_len}")
  print(f"  Total tokens: {total_tokens}, distribution: {distribution}")
  print(f"  Heads: Q={num_q_heads}, KV={num_kv_heads}, Dim={head_dim}")
  print(
      f"  Tiling: bq_sz={FLAGS.bq_sz}, bkv_sz={FLAGS.bkv_sz},"
      f" bq_c_sz={FLAGS.bq_c_sz}"
  )
  print(f"  RoPE: theta={theta}, ordering={ordering}")
  print("=" * 85)

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

  # Generate inputs
  key = jax.random.PRNGKey(42)
  k1, k2, k3, k4 = jax.random.split(key, 4)

  q = jax.random.normal(
      k1, (total_tokens, num_q_heads, head_dim), dtype=jnp.bfloat16
  )
  k = jax.random.normal(
      k2, (total_tokens, num_kv_heads, head_dim), dtype=jnp.float8_e4m3fn
  )
  v = jax.random.normal(
      k3, (total_tokens, num_kv_heads, head_dim), dtype=jnp.float8_e4m3fn
  )

  cache_shape = wrapper.get_kv_cache_shape(
      num_pages,
      page_size,
      num_kv_heads,
      head_dim,
      kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
      kv_dtype=jnp.float8_e4m3fn,
  )
  kv_cache = jax.random.normal(k4, cache_shape, dtype=jnp.float8_e4m3fn)

  max_model_len = 4096
  pages_per_seq = max_model_len // page_size

  cu_q_lens_list = [0]
  for _ in range(num_requests):
    cu_q_lens_list.append(cu_q_lens_list[-1] + q_len)
  kv_lens_list = [kv_len] * num_requests

  cu_q_lens = jnp.pad(
      jnp.array(cu_q_lens_list, dtype=jnp.int32),
      (0, max_num_seqs + 1 - len(cu_q_lens_list)),
  )
  kv_lens = jnp.pad(
      jnp.array(kv_lens_list, dtype=jnp.int32),
      (0, max_num_seqs - len(kv_lens_list)),
  )
  page_indices = jnp.arange(max_num_seqs * pages_per_seq, dtype=jnp.int32)

  # 1a. Standalone / No-RoPE RPA Step (swapaxes)
  def no_rope_step(q_in, k_in, v_in, cache_in):
    out, new_cache = wrapper.ragged_paged_attention(
        queries=q_in,
        keys=k_in,
        values=v_in,
        kv_cache=cache_in,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        sm_scale=sm_scale,
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
        decode_block_sizes=decode_blocks,
        prefill_block_sizes=prefill_blocks,
        vmem_limit_bytes=vmem_limit_bytes,
        apply_rope=False,
        use_strided_dma=False,
    )
    return out, new_cache

  # 1b. Standalone / No-RoPE RPA Step (Strided DMA, Zero-Copy)
  def no_rope_strided_dma_step(q_in, k_in, v_in, cache_in):
    out, new_cache = wrapper.ragged_paged_attention(
        queries=q_in,
        keys=k_in,
        values=v_in,
        kv_cache=cache_in,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        sm_scale=sm_scale,
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
        decode_block_sizes=decode_blocks,
        prefill_block_sizes=prefill_blocks,
        vmem_limit_bytes=vmem_limit_bytes,
        apply_rope=False,
        use_strided_dma=True,
    )
    return out, new_cache

  # 2a. Baseline Step (Standalone JIT RoPE + Standard RPA with swapaxes)
  def baseline_step(q_in, k_in, v_in, cache_in):
    q_rot = apply_reference_rope(q_in, theta=theta, ordering=ordering)
    k_rot = apply_reference_rope(k_in, theta=theta, ordering=ordering)
    out, new_cache = wrapper.ragged_paged_attention(
        queries=q_rot,
        keys=k_rot,
        values=v_in,
        kv_cache=cache_in,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        sm_scale=sm_scale,
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
        decode_block_sizes=decode_blocks,
        prefill_block_sizes=prefill_blocks,
        vmem_limit_bytes=vmem_limit_bytes,
        apply_rope=False,
        use_strided_dma=False,
    )
    return out, new_cache

  # 2b. Standalone RoPE + Strided DMA RPA (No In-Kernel RoPE, Zero-Copy DMA)
  def baseline_strided_dma_step(q_in, k_in, v_in, cache_in):
    q_rot = apply_reference_rope(q_in, theta=theta, ordering=ordering)
    k_rot = apply_reference_rope(k_in, theta=theta, ordering=ordering)
    out, new_cache = wrapper.ragged_paged_attention(
        queries=q_rot,
        keys=k_rot,
        values=v_in,
        kv_cache=cache_in,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        sm_scale=sm_scale,
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
        decode_block_sizes=decode_blocks,
        prefill_block_sizes=prefill_blocks,
        vmem_limit_bytes=vmem_limit_bytes,
        apply_rope=False,
        use_strided_dma=True,
    )
    return out, new_cache

  # 3. Fused Step (In-Kernel RoPE bRPA - Token-Major with swapaxes)
  def fused_step(q_in, k_in, v_in, cache_in):
    out, new_cache = wrapper.ragged_paged_attention_rope(
        queries=q_in,
        keys=k_in,
        values=v_in,
        kv_cache=cache_in,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        rope_theta=theta,
        rope_dim=head_dim,
        rope_input_ordering=ordering,
        sm_scale=sm_scale,
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
        decode_block_sizes=decode_blocks,
        prefill_block_sizes=prefill_blocks,
        vmem_limit_bytes=vmem_limit_bytes,
        is_kv_group_major=False,
        use_strided_dma=False,
    )
    return out, new_cache

  # 4. Fused Strided DMA Step (In-Kernel RoPE bRPA - Token-Major Strided DMA, Zero-Copy)
  def strided_dma_step(q_in, k_in, v_in, cache_in):
    out, new_cache = wrapper.ragged_paged_attention_rope(
        queries=q_in,
        keys=k_in,
        values=v_in,
        kv_cache=cache_in,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        rope_theta=theta,
        rope_dim=head_dim,
        rope_input_ordering=ordering,
        sm_scale=sm_scale,
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
        decode_block_sizes=decode_blocks,
        prefill_block_sizes=prefill_blocks,
        vmem_limit_bytes=vmem_limit_bytes,
        is_kv_group_major=False,
        use_strided_dma=True,
    )
    return out, new_cache

  # 5. KV-Group Major Step (In-Kernel RoPE bRPA - Offline Permuted Q, Zero-Copy)
  g = num_q_heads // num_kv_heads
  q_kv_major = (
      q.reshape(total_tokens, num_kv_heads, g, head_dim).transpose(1, 0, 2, 3)
  )

  def kv_group_major_step(q_in, k_in, v_in, cache_in):
    out, new_cache = wrapper.ragged_paged_attention_rope(
        queries=q_in,
        keys=k_in,
        values=v_in,
        kv_cache=cache_in,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        rope_theta=theta,
        rope_dim=head_dim,
        rope_input_ordering=ordering,
        sm_scale=sm_scale,
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
        decode_block_sizes=decode_blocks,
        prefill_block_sizes=prefill_blocks,
        vmem_limit_bytes=vmem_limit_bytes,
        is_kv_group_major=True,
        use_strided_dma=False,
    )
    return out, new_cache

  jitted_no_rope = jax.jit(no_rope_step, donate_argnums=(3,))
  jitted_no_rope_sd = jax.jit(no_rope_strided_dma_step, donate_argnums=(3,))
  jitted_baseline = jax.jit(baseline_step, donate_argnums=(3,))
  jitted_baseline_sd = jax.jit(baseline_strided_dma_step, donate_argnums=(3,))
  jitted_fused = jax.jit(fused_step, donate_argnums=(3,))
  jitted_strided_dma = jax.jit(strided_dma_step, donate_argnums=(3,))
  jitted_kv_group_major = jax.jit(kv_group_major_step, donate_argnums=(3,))

  # Run warmup
  print("\nRunning warmup iterations...", flush=True)
  out_nr, kv_cache = jitted_no_rope(q, k, v, kv_cache)
  out_nr.block_until_ready()

  out_nrsd, kv_cache = jitted_no_rope_sd(q, k, v, kv_cache)
  out_nrsd.block_until_ready()

  out_b, kv_cache = jitted_baseline(q, k, v, kv_cache)
  out_b.block_until_ready()

  out_bsd, kv_cache = jitted_baseline_sd(q, k, v, kv_cache)
  out_bsd.block_until_ready()

  out_f, kv_cache = jitted_fused(q, k, v, kv_cache)
  out_f.block_until_ready()

  out_sd, kv_cache = jitted_strided_dma(q, k, v, kv_cache)
  out_sd.block_until_ready()

  out_kv, kv_cache = jitted_kv_group_major(q_kv_major, k, v, kv_cache)
  out_kv.block_until_ready()

  for _ in range(FLAGS.num_warmup):
    out_b, kv_cache = jitted_baseline(q, k, v, kv_cache)
    out_b.block_until_ready()
    out_bsd, kv_cache = jitted_baseline_sd(q, k, v, kv_cache)
    out_bsd.block_until_ready()
    out_f, kv_cache = jitted_fused(q, k, v, kv_cache)
    out_f.block_until_ready()
    out_sd, kv_cache = jitted_strided_dma(q, k, v, kv_cache)
    out_sd.block_until_ready()
    out_kv, kv_cache = jitted_kv_group_major(q_kv_major, k, v, kv_cache)
    out_kv.block_until_ready()

  # Numerical parity
  diff_bsd = jnp.abs(out_bsd - out_b)
  max_diff_bsd = float(jnp.max(diff_bsd))
  mean_diff_bsd = float(jnp.mean(diff_bsd))
  print("\nNumerical Parity Verification on Hardware:")
  print("  Standalone RoPE + Strided DMA vs Baseline (swapaxes):")
  print(f"    Max Absolute Difference:  {max_diff_bsd:.6e}")
  print(f"    Mean Absolute Difference: {mean_diff_bsd:.6e}")

  diff = jnp.abs(out_f - out_b)
  max_diff = float(jnp.max(diff))
  mean_diff = float(jnp.mean(diff))
  print("  Token-Major Fused (swapaxes) vs Baseline:")
  print(f"    Max Absolute Difference:  {max_diff:.6e}")
  print(f"    Mean Absolute Difference: {mean_diff:.6e}")

  diff_sd = jnp.abs(out_sd - out_b)
  max_diff_sd = float(jnp.max(diff_sd))
  mean_diff_sd = float(jnp.mean(diff_sd))
  print("  Token-Major Strided DMA (Fused RoPE) vs Baseline:")
  print(f"    Max Absolute Difference:  {max_diff_sd:.6e}")
  print(f"    Mean Absolute Difference: {mean_diff_sd:.6e}")

  # out_kv is [H_kv, T, G, D], compare against out_b [T, H_q, D]
  out_kv_reconstructed = (
      out_kv.transpose(1, 0, 2, 3).reshape(total_tokens, num_q_heads, head_dim)
  )
  diff_kv = jnp.abs(out_kv_reconstructed - out_b)
  max_diff_kv = float(jnp.max(diff_kv))
  mean_diff_kv = float(jnp.mean(diff_kv))
  print("  KV-Group Major Zero-Copy vs Baseline:")
  print(f"    Max Absolute Difference:  {max_diff_kv:.6e}")
  print(f"    Mean Absolute Difference: {mean_diff_kv:.6e}")

  def time_func(fn, inputs, label):
    session = None
    session_id = None
    if FLAGS.record_xprof and xprof_session:
      try:
        session = xprof_session.XprofSession()
        session.start_session(
            trace_mode="TRACE_COMPUTE_AND_DMA",
            enable_fw_throttle_event=True,
            enable_fw_power_level_event=True,
            enable_fw_thermal_event=True,
            power_trace_level="POWER_TRACE_NORMAL",
        )
      except Exception as e:
        print(f"Warning starting XProf for {label}: {e}")

    curr_inputs = list(inputs)
    latencies = []
    for _ in range(FLAGS.num_iters):
      t0 = time.perf_counter()
      out, new_cache = fn(*curr_inputs)
      out.block_until_ready()
      t1 = time.perf_counter()
      curr_inputs[3] = new_cache
      latencies.append((t1 - t0) * 1e6)

    if session:
      try:
        session_id = session.end_session_and_get_session_id()
      except Exception as e:
        print(f"Warning stopping XProf for {label}: {e}")

    return np.array(latencies), session_id, curr_inputs[3]

  print("\n" + "=" * 85)
  print("BENCHMARK EXECUTION RESULTS (4 reqs x 1024 tokens = 4096 tokens)")
  print("=" * 85)

  # 1a. No RoPE (swapaxes)
  no_rope_lats, no_rope_sid, kv_cache = time_func(
      jitted_no_rope, (q, k, v, kv_cache), "no_rope_swapaxes"
  )
  no_rope_durs = get_kernel_durations(no_rope_sid) if no_rope_sid else {}
  no_rope_device_us = (
      np.mean(no_rope_durs["RPAm"]) if no_rope_durs.get("RPAm") else 0.0
  )
  print("1a. bRPA Standalone (No RoPE, swapaxes):")
  print(f"    Host Mean Latency:   {np.mean(no_rope_lats):.2f} µs")
  if no_rope_device_us > 0:
    print(f"    Device (RPAm) Mean:  {no_rope_device_us:.2f} µs")
  if no_rope_sid:
    print(f"    XProf Trace:         http://xprof/?session_id={no_rope_sid}")

  # 1b. No RoPE (Strided DMA)
  no_rope_sd_lats, no_rope_sd_sid, kv_cache = time_func(
      jitted_no_rope_sd, (q, k, v, kv_cache), "no_rope_strided_dma"
  )
  no_rope_sd_durs = (
      get_kernel_durations(no_rope_sd_sid) if no_rope_sd_sid else {}
  )
  no_rope_sd_device_us = (
      np.mean(no_rope_sd_durs["RPAm"]) if no_rope_sd_durs.get("RPAm") else 0.0
  )
  print("\n1b. bRPA Standalone (No RoPE, Strided DMA Zero-Copy):")
  print(f"    Host Mean Latency:   {np.mean(no_rope_sd_lats):.2f} µs")
  if no_rope_sd_device_us > 0:
    print(f"    Device (RPAm) Mean:  {no_rope_sd_device_us:.2f} µs")
  if no_rope_sd_sid:
    print(f"    XProf Trace:         http://xprof/?session_id={no_rope_sd_sid}")

  # 2a. Baseline (Standalone RoPE + bRPA with swapaxes)
  base_lats, base_sid, kv_cache = time_func(
      jitted_baseline, (q, k, v, kv_cache), "baseline_swapaxes"
  )
  base_durs = get_kernel_durations(base_sid) if base_sid else {}
  base_device_us = (
      np.mean(base_durs["RPAm"]) if base_durs.get("RPAm") else 0.0
  )
  print("\n2a. Baseline (Standalone RoPE + bRPA with swapaxes):")
  print(f"    Host Mean Latency:   {np.mean(base_lats):.2f} µs")
  if base_device_us > 0:
    print(f"    Device (RPAm) Mean:  {base_device_us:.2f} µs")
  if base_sid:
    print(f"    XProf Trace:         http://xprof/?session_id={base_sid}")

  # 2b. Standalone RoPE + Strided DMA (No In-Kernel RoPE, Zero-Copy DMA)
  base_sd_lats, base_sd_sid, kv_cache = time_func(
      jitted_baseline_sd, (q, k, v, kv_cache), "standalone_rope_strided_dma"
  )
  base_sd_durs = (
      get_kernel_durations(base_sd_sid) if base_sd_sid else {}
  )
  base_sd_device_us = (
      np.mean(base_sd_durs["RPAm"]) if base_sd_durs.get("RPAm") else 0.0
  )
  print("\n2b. Standalone RoPE + Strided DMA (No In-Kernel RoPE, Zero-Copy):")
  print(f"    Host Mean Latency:   {np.mean(base_sd_lats):.2f} µs")
  if base_sd_device_us > 0:
    print(f"    Device (RPAm) Mean:  {base_sd_device_us:.2f} µs")
  if base_sd_sid:
    print(f"    XProf Trace:         http://xprof/?session_id={base_sd_sid}")

  # 3. Fused (In-Kernel RoPE + bRPA - Token-Major with swapaxes)
  fused_lats, fused_sid, kv_cache = time_func(
      jitted_fused, (q, k, v, kv_cache), "fused_swapaxes"
  )
  fused_durs = get_kernel_durations(fused_sid) if fused_sid else {}
  fused_device_us = (
      np.mean(fused_durs["RPAm"]) if fused_durs.get("RPAm") else 0.0
  )
  print("\n3. Optimized Fused (In-Kernel RoPE bRPA - Token Major swapaxes):")
  print(f"    Host Mean Latency:   {np.mean(fused_lats):.2f} µs")
  if fused_device_us > 0:
    print(f"    Device (RPAm) Mean:  {fused_device_us:.2f} µs")
  if fused_sid:
    print(f"    XProf Trace:         http://xprof/?session_id={fused_sid}")

  # 4. Fused Strided DMA (In-Kernel RoPE + bRPA - Token-Major Strided DMA, Zero-Copy)
  sd_lats, sd_sid, kv_cache = time_func(
      jitted_strided_dma, (q, k, v, kv_cache), "fused_strided_dma"
  )
  sd_durs = get_kernel_durations(sd_sid) if sd_sid else {}
  sd_device_us = np.mean(sd_durs["RPAm"]) if sd_durs.get("RPAm") else 0.0
  print("\n4. Optimized Strided DMA (In-Kernel RoPE + Strided DMA Zero-Copy):")
  print(f"    Host Mean Latency:   {np.mean(sd_lats):.2f} µs")
  if sd_device_us > 0:
    print(f"    Device (RPAm) Mean:  {sd_device_us:.2f} µs")
  if sd_sid:
    print(f"    XProf Trace:         http://xprof/?session_id={sd_sid}")

  # 5. KV-Group Major (Zero-Copy Q + In-Kernel RoPE bRPA)
  kv_lats, kv_sid, kv_cache = time_func(
      jitted_kv_group_major, (q_kv_major, k, v, kv_cache), "fused_kv_group_major"
  )
  kv_durs = get_kernel_durations(kv_sid) if kv_sid else {}
  kv_device_us = np.mean(kv_durs["RPAm"]) if kv_durs.get("RPAm") else 0.0
  print("\n5. Optimized Zero-Copy (KV-Group Major Q + In-Kernel RoPE bRPA):")
  print(f"    Host Mean Latency:   {np.mean(kv_lats):.2f} µs")
  if kv_device_us > 0:
    print(f"    Device (RPAm) Mean:  {kv_device_us:.2f} µs")
  if kv_sid:
    print(f"    XProf Trace:         http://xprof/?session_id={kv_sid}")

  print("\n" + "=" * 85)
  print("COMPARISON & SPEEDUP SUMMARY")
  print("=" * 85)
  speedup_base_sd = np.mean(base_lats) / max(np.mean(base_sd_lats), 1e-6)
  saved_base_sd_us = np.mean(base_lats) - np.mean(base_sd_lats)
  print(
      f"Standalone RoPE + Strided DMA Savings: {saved_base_sd_us:+.2f} µs"
      f" ({speedup_base_sd:.2f}x speedup vs Baseline)"
  )

  speedup_fused = np.mean(base_lats) / max(np.mean(fused_lats), 1e-6)
  saved_fused_us = np.mean(base_lats) - np.mean(fused_lats)
  print(
      f"Token-Major Fused (swapaxes) Savings:  {saved_fused_us:+.2f} µs"
      f" ({speedup_fused:.2f}x speedup vs Baseline)"
  )

  speedup_sd = np.mean(base_lats) / max(np.mean(sd_lats), 1e-6)
  saved_sd_us = np.mean(base_lats) - np.mean(sd_lats)
  print(
      f"Token-Major Strided DMA (Fused) Sav.:  {saved_sd_us:+.2f} µs"
      f" ({speedup_sd:.2f}x speedup vs Baseline)"
  )

  speedup_kv = np.mean(base_lats) / max(np.mean(kv_lats), 1e-6)
  saved_kv_us = np.mean(base_lats) - np.mean(kv_lats)
  print(
      f"Zero-Copy KV-Group Major Savings:     {saved_kv_us:+.2f} µs"
      f" ({speedup_kv:.2f}x speedup vs Baseline)"
  )

  print("-" * 85)
  fused_rope_vs_standalone_sd = np.mean(base_sd_lats) - np.mean(sd_lats)
  print(
      f"Pure In-Kernel RoPE Fusion Savings (under Strided DMA): {fused_rope_vs_standalone_sd:+.2f} µs"
  )
  print("=" * 85)


def main(argv):
  del argv
  run_benchmark()


if __name__ == "__main__":
  app.run(main)
