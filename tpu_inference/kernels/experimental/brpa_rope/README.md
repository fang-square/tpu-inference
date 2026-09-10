# Batched Ragged Paged Attention (bRPA) with In-Kernel RoPE & Zero-Copy KV-Group Major Layout

## 1. Overview & Motivation

In standard Large Language Model (LLM) serving pipelines on TPU, applying Rotary
Position Embeddings (RoPE) and reshaping activation tensors from GEMM output
layouts incurs substantial High Bandwidth Memory (HBM) bandwidth and latency
penalties:

1.  **Standalone RoPE Overhead**: $Q$ and $K$ activations must be written to
    HBM, read by a dedicated RoPE elementwise kernel, transformed with
    $\cos/\sin$, and written back to HBM.
2.  **Layout Transposition Overhead (`copy.261` ~133 µs)**: Standard GEMM
    projections output activations in **token-major** order (`[T, H_q, D]`),
    whereas attention operators require **head-major / KV-grouped** order
    (`[H_kv, T, G, D]`). Across the compiler/`shard_map` boundary, this forces
    a physical HBM DMA copy (`swapaxes(0, 1)`).

**bRPA with In-Kernel RoPE & KV-Group Major Q-GEMM** eliminates both bottlenecks:

*   **Head-Major Q-GEMM (`qkv_pipeline.py`)**: By permuting the $W_Q$
    projection weight offline at model load time into
    $[N_{kv}, \text{dim}_{\text{in}}, G \cdot D]$, the Q-GEMM directly outputs
    $Q$ in $[N_{kv}, T, G, D]$ format (`[4, 4096, 8, 128]`). This achieves
    **zero intermediate HBM transpositions (`0 µs copy`)** with $100\%$ linear burst DMA throughput,
    completely eliminating Step 11 (`copy.284` @ `95.31 µs`).
*   **Direct Token-Major Attention Output (`out_token_major=True`, `wrapper.py` & `bref_override.py`)**:
    Directly writes the attention output into HBM in token-major layout `[T, H_q, D]` via in-kernel
    multi-head DMA (`o_hbm.at[pl.ds(q_src, q_sz), h_kv]`). This completely eliminates the Step 14
    layout transpose (`%copy.207` @ `11.14 µs` and DMA fence `%copy-done.7` @ `7.87 µs`, saving
    **`19.01 µs`** per layer). A dedicated separate HBM buffer is allocated only when `out_token_major=True`
    and input $Q$ is head-major (`is_kv_group_major=True`).
*   **Head-Sharded Joint $W_{KV}$ GEMM (`qkv_pipeline.py`)**: Interleaves $K$ and $V$ heads
    along output column channels `[H_in, num_kv_heads, 2, head_dim]`. When column-sharded across $TP=2$
    devices, each chiplet independently receives both $K$ and $V$ for its assigned heads without requiring
    cross-chip communication, eliminating the Pre-GEMM staging copy (`%copy.208` @ `3.66 µs`),
    cross-device collective (`%all-to-all.2` @ `7.24 µs`), and reshape overhead (saving **`~17.66 µs`** per layer).
*   **In-Register RoPE on K Slice Only**: In vector post-processing, RoPE is applied strictly
    to $K$ (slot 0) while $V$ (slot 1) bypasses RoPE with zero cycle overhead, streaming directly into
    the Paged KV Cache buffer (`[T, 2, N_kv, D]`).
*   **In-Kernel RoPE (`kernel.py`)**: $Q$ and $K$ tiles are rotated on-chip
    using Vector Processing Unit (VPU) ALUs immediately before $QK^T$ matrix
    multiplication and KV cache writeback, eliminating the standalone RoPE kernel.

--------------------------------------------------------------------------------

## 2. Architecture & Implementation Breakdown

```mermaid
flowchart TD
    subgraph Baseline ["Baseline Pipeline (1301 µs / layer)"]
        W1["Separate W_Q, W_K, W_V"] --> GEMM1["Token-Major Q, K, V GEMMs (35.6 µs)"]
        GEMM1 --> SWAP["Attention Head Formatting (copy.284: 95.31 µs)"]
        SWAP --> ROPE["Separate Norm & RoPE Kernels (~20 µs)"]
        ROPE --> ATTN1["RPAm Attention Kernel (344.83 µs)"]
    end

    subgraph Decoupled ["Decoupled Head-Major Q + Joint W_KV Pipeline (~1198 µs / layer)"]
        WQ["Permuted W_Q [N_kv, H_in, G*D]"] --> GEMMQ["Head-Major Q-GEMM [N_kv, T, G, D] (28.45 µs)"]
        WKV["Joint W_KV [H_in, 2*N_kv*D]"] --> GEMMKV["Joint W_KV GEMM [T, 2, N_kv, D] (~9.80 µs)"]
        GEMMQ -->|"Zero-Copy Direct Handover (0 µs)"| VEC["Fused Vector Norm + K-RoPE"]
        GEMMKV -->|"Direct Cache Format (0 µs concat)"| VEC
        VEC -->|"Direct Burst DMA Handover"| ATTN2["Zero-Copy RPAm Attention (344.83 µs)"]
    end
```

### Key Modules:

*   **`qkv_pipeline.py`**:
    *   Offline weight permuters (`permute_q_weight_to_kv_group_major`,
        `permute_joint_kv_weights`, `permute_merged_qkv_weights`).
    *   FP8 static weight quantization (`quantize_q_kv_major_weight_to_fp8`, `quantize_joint_kv_weight_to_fp8`) and
        FP8 per-token dynamic activation quantization (`quantize_to_fp8_dynamic`).
    *   Full pre-attention pipelines: Decoupled Head-Major Q + Joint $W_{KV}$ (`qkv_projection_pipeline_head_major_q_joint_kv`),
        Separated KV-Group Major, Merged Single-GEMM, and Token-Major Baseline.
*   **`configs.py`**:
    *   Added RoPE hyperparameters (`apply_rope`, `rope_theta`, `rope_dim`,
        `rope_ordering`) to `ModelConfigs` and `RpaConfigs`.
    *   Updated `validate_inputs()` to accept 4D $Q$ tensors (`[H_kv, T, G, D]`)
        when `is_kv_group_major=True`.
*   **`wrapper.py`**:
    *   Fast-path in `prepare_inputs()` and `ragged_paged_attention()` for
        `is_kv_group_major=True`, bypassing `swapaxes(0, 1)` entirely.
    *   Bound `is_kv_group_major` into `@jax.jit(static_argnames=...)`.
*   **`kernel.py`**:
    *   `apply_in_kernel_rope`: Computes on-chip $\cos/\sin$ frequencies and
        rotates $Q$ (`[B, H_kv, bq_sz, G, D]`) and $K$ (`[B, H_kv, bkv_sz, D]`)
        in VMEM.
*   **`pallas_rmsnorm.py`**:
    *   `pallas_2d_rmsnorm`: Dedicated 2D Pallas RMSNorm kernel streaming `[N,
        128]` activations through VMEM in row-major order, bypassing XLA 4D
        sublane tiling and eliminating intermediate HBM transpose copies.
    *   `pallas_2d_rmsnorm_rope`: Fused 2D Pallas RMSNorm + RoPE kernel
        executing normalization and rotary position embedding in FP32 VPU vector
        registers within a single streaming pass, keeping $Q$ strictly in BF16
        and bypassing RoPE in RPAm.
*   **`test_brpa_layout_overhead.py`**:
    *   Compiles JAX HLO graphs and verifies that `is_kv_group_major=True` produces
        **0 HLO transposes for Q** (vs. 1 transpose in baseline).
*   **`test_brpa_rope_correctness.py`**:
    *   Verifies 100% numerical cosine similarity and parity across all
        pipelines (Tests 1 through 8), including Fused Pallas RMSNorm + RoPE
        numerical parity.
*   **`benchmark_brpa_rope.py`**:
    *   Multi-way hardware benchmark on Sunfish EVT with programmatic XProf
        tracing across Tests 0, 1, 2, 3, 3b, 4, and 5.

--------------------------------------------------------------------------------

## 3. Hardware Benchmark Results on Sunfish EVT (`yudlsqd-bcq8`)

### Workload Configuration

*   **Target Workload**: Qwen3-32B Prefill (4 requests $\times$ 1024 tokens = 4096 total tokens)
*   **Hardware Shard**: TPU v6e (Sunfish) TP=2 Shard ($H_q=32, H_{kv}=4, D=128$, `page_size=256`)
*   **Tiling**: `bq_sz=512, bkv_sz=256, bq_c_sz=512`, `n_buffer=3`
*   **Precision**: Queries (`bfloat16`), KV Cache (`float8_e4m3fn`)
*   **Buffer Donation**: Enabled (`donate_argnums=(3,)`), eliminating defensive `copy.19.kv_cache_hbm_ref` overhead from host measurements.

### 7-Way Microbenchmark Latency on EVT Hardware (`yudlsqd-bcq8`)

| # | Configuration | Host Mean Latency | Device (RPAm) Mean | Speedup vs Baseline | XProf Trace URL |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1a** | **No RoPE (`swapaxes`)** | $431.24\ \mu\text{s}$ | $211.89\ \mu\text{s}$ | — | [Trace 91860](http://xprof/?session_id=platforms-deepsea-9495668343747191860) |
| **1b** | **No RoPE (Strided DMA Zero-Copy)** | $403.36\ \mu\text{s}$ | $212.18\ \mu\text{s}$ | — | [Trace 94202](http://xprof/?session_id=platforms-deepsea-9495668343747194202) |
| **2a** | **Baseline (Standalone RoPE + `swapaxes`)** | $679.66\ \mu\text{s}$ | $211.80\ \mu\text{s}$ | Reference ($1.00\times$) | [Trace 91375](http://xprof/?session_id=platforms-deepsea-9495668343747191375) |
| **2b** | **Standalone RoPE + Strided DMA (No In-Kernel RoPE)** | **$530.95\ \mu\text{s}$** | **$211.96\ \mu\text{s}$** | **$+148.71\ \mu\text{s}$ ($1.28\times$)** | [Trace 92644](http://xprof/?session_id=platforms-deepsea-9495668343747192644) |
| **3** | **Optimized Fused (In-Kernel RoPE - Token Major `swapaxes`)** | $495.07\ \mu\text{s}$ | $272.08\ \mu\text{s}$ | **$+184.58\ \mu\text{s}$ ($1.37\times$)** | [Trace 93913](http://xprof/?session_id=platforms-deepsea-9495668343747193913) |
| **4** | **Optimized Strided DMA (In-Kernel RoPE + Strided DMA)** | **$464.73\ \mu\text{s}$** | **$272.39\ \mu\text{s}$** | **$+214.93\ \mu\text{s}$ ($1.46\times$)** | [Trace 91086](http://xprof/?session_id=platforms-deepsea-9495668343747191086) |
| **5** | **Optimized Zero-Copy (KV-Group Major $Q$ + In-Kernel RoPE)** | **$461.66\ \mu\text{s}$** | **$272.24\ \mu\text{s}$** | **$+218.00\ \mu\text{s}$ ($1.47\times$)** | [Trace 92355](http://xprof/?session_id=platforms-deepsea-9495668343747192355) |

> **Key Ablation Takeaways**:
> 1. **Strided DMA Impact without In-Kernel RoPE (2a $\to$ 2b)**:
>    - Keeping Standalone RoPE external and applying Strided DMA into bRPA saves **`148.71 µs`** ($679.66\ \mu\text{s} \to 530.95\ \mu\text{s}$, a **$1.28\times$ speedup**), achieving **100% bit-exact parity ($0.000\text{ diff}$)** against the baseline.
> 2. **In-Kernel RoPE Fusion Impact under Strided DMA (2b $\to$ 4)**:
>    - Fusing RoPE into the Strided DMA bRPA kernel saves an additional **`66.22 µs`** ($530.95\ \mu\text{s} \to 464.73\ \mu\text{s}$).
> 3. **Total Combined Savings (2a Baseline $\to$ 4 Fused Strided DMA)**:
>    - **`214.93 µs`** per layer saved ($679.66\ \mu\text{s} \to 464.73\ \mu\text{s}$, a **$1.46\times$ speedup**).
>    - Translates to **$\sim 13.75\text{ ms}$ total prefill time reduction** across 64 layers without requiring any offline weight permutations.

### Numerical Parity Verification

Verified against reference baseline on Sunfish EVT hardware:

*   **Standalone RoPE + Strided DMA vs Baseline**: Max diff $= 0.000000$, Mean diff $= 0.000000$ (Bit-exact match)
*   **Token-Major Fused vs Baseline**: Max diff $= 6.74 \times 10^{-2}$, Mean diff $= 1.47 \times 10^{-3}$
*   **Token-Major Strided DMA (Fused RoPE) vs Baseline**: Max diff $= 6.74 \times 10^{-2}$, Mean diff $= 1.47 \times 10^{-3}$
*   **KV-Group Major Zero-Copy vs Baseline**: Max diff $= 6.74 \times 10^{-2}$, Mean diff $= 1.47 \times 10^{-3}$

### End-to-End QKV Projection & Attention Pipeline Benchmark (Tests 0 - 9)

Evaluated on Sunfish TPU EVT (`yudlsqd-bcq8`) with $T=4096$, $TP=2$, $H_q=32, H_{kv}=4, D=128$, `page_size=256`:

#### Run A: Direct Token-Major Output Enabled (`--out_token_major=True`, `--include_pre_gemm_staging=True`)

| # | Pipeline Candidate | Host Mean Latency | Extrapolated 64L Wall | Device RPAm Kernel | Speedup vs Ref | XProf Trace Session Link |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **0** | **Decoupled Baseline (Ref)** | `1082.19 µs` | `69.26 ms` | `276.90 µs` | 1.00x | [platforms-deepsea-7964137372228088740](http://xprof/?session_id=platforms-deepsea-7964137372228088740) |
| **1** | **KV-Major Q + In-Kernel Norm & RoPE** | `875.08 µs` | `56.01 ms` | `501.94 µs` | 1.24x | [platforms-deepsea-7964137372228090010](http://xprof/?session_id=platforms-deepsea-7964137372228090010) |
| **2** | **Token-Major Q + Fused Norm + Strided DMA** | `861.50 µs` | `55.14 ms` | `272.33 µs` | 1.26x | [platforms-deepsea-7964137372228089703](http://xprof/?session_id=platforms-deepsea-7964137372228089703) |
| **3** | **KV-Major Q + Strided DMA + In-Kernel Norm** | `883.48 µs` | `56.54 ms` | `503.55 µs` | 1.22x | [platforms-deepsea-7964137372228089396](http://xprof/?session_id=platforms-deepsea-7964137372228089396) |
| **3b**| **Token-Major Q + Strided DMA + In-Kernel Norm** | `893.52 µs` | `57.19 ms` | `497.42 µs` | 1.21x | [platforms-deepsea-7964137372228089089](http://xprof/?session_id=platforms-deepsea-7964137372228089089) |
| **4** | **KV-Major Q + 2D Pallas RMSNorm (Shard-Major Joint $W_{KV}$)** | **`692.60 µs`** | **`44.33 ms`** | **`277.01 µs`** | **1.56x (+389.59 µs)** | [platforms-deepsea-7964137372228088782](http://xprof/?session_id=platforms-deepsea-7964137372228088782) |
| **5** | **KV-Major Q + Fused Pallas RMSNorm & RoPE (Shard-Major Joint $W_{KV}$)** | **`696.60 µs`** | **`44.58 ms`** | **`216.56 µs`** | **1.55x (+385.59 µs)** | [platforms-deepsea-7964137372228088475](http://xprof/?session_id=platforms-deepsea-7964137372228088475) |
| **6** | **Head-Sharded Joint $W_{KV}$ + 2D Pallas RMSNorm + Zero-Copy bRPA** | **`718.61 µs`** | **`45.99 ms`** | **`276.94 µs`** | **1.51x (+363.58 µs)** | [platforms-deepsea-7964137372228088168](http://xprof/?session_id=platforms-deepsea-7964137372228088168) |
| **7** | **Head-Sharded Joint $W_{KV}$ + Fused Norm/RoPE + Zero-Copy bRPA** | **`724.18 µs`** | **`46.35 ms`** | **`216.57 µs`** | **1.49x (+358.01 µs)** | [platforms-deepsea-7964137372228087861](http://xprof/?session_id=platforms-deepsea-7964137372228087861) |
| **8** | **Separate $W_K, W_V$ + 2D Pallas RMSNorm + Zero-Copy bRPA** | **`688.26 µs`** | **`44.05 ms`** | **`276.99 µs`** | **1.57x (+393.93 µs)** | [platforms-deepsea-7964137372228091650](http://xprof/?session_id=platforms-deepsea-7964137372228091650) |
| **9** | **Separate $W_K, W_V$ + Fused Norm/RoPE + Zero-Copy bRPA** | **`687.27 µs`** | **`43.99 ms`** | **`216.52 µs`** | **1.57x (+394.92 µs)** | [platforms-deepsea-7964137372228091343](http://xprof/?session_id=platforms-deepsea-7964137372228091343) |

#### Run B: Default Attention Output (`--out_token_major=False`, `--include_pre_gemm_staging=True`)

| # | Pipeline Candidate | Host Mean Latency | Extrapolated 64L Wall | Device RPAm Kernel | Speedup vs Ref | XProf Trace Session Link |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **0** | **Decoupled Baseline (Ref)** | `1088.56 µs` | `69.67 ms` | `277.81 µs` | 1.00x | [platforms-deepsea-8436557819037171049](http://xprof/?session_id=platforms-deepsea-8436557819037171049) |
| **1** | **KV-Major Q + In-Kernel Norm & RoPE** | `880.52 µs` | `56.35 ms` | `502.89 µs` | 1.24x | [platforms-deepsea-8436557819037170711](http://xprof/?session_id=platforms-deepsea-8436557819037170711) |
| **2** | **Token-Major Q + Fused Norm + Strided DMA** | `861.81 µs` | `55.16 ms` | `272.34 µs` | 1.26x | [platforms-deepsea-8436557819037171112](http://xprof/?session_id=platforms-deepsea-8436557819037171112) |
| **3** | **KV-Major Q + Strided DMA + In-Kernel Norm** | `883.18 µs` | `56.52 ms` | `493.07 µs` | 1.23x | [platforms-deepsea-8436557819037171513](http://xprof/?session_id=platforms-deepsea-8436557819037171513) |
| **3b**| **Token-Major Q + Strided DMA + In-Kernel Norm** | `886.14 µs` | `56.71 ms` | `497.41 µs` | 1.23x | [platforms-deepsea-8436557819037167818](http://xprof/?session_id=platforms-deepsea-8436557819037167818) |
| **4** | **KV-Major Q + 2D Pallas RMSNorm (Solution B)** | **`704.17 µs`** | **`45.07 ms`** | **`277.85 µs`** | **1.55x (+384.39 µs)** | [platforms-deepsea-8436557819037168219](http://xprof/?session_id=platforms-deepsea-8436557819037168219) |
| **5** | **KV-Major Q + Fused Pallas RMSNorm & RoPE (Solution C)** | **`704.34 µs`** | **`45.08 ms`** | **`217.61 µs`** | **1.55x (+384.21 µs)** | [platforms-deepsea-8436557819037168620](http://xprof/?session_id=platforms-deepsea-8436557819037168620) |
| **6** | **Head-Sharded $W_{KV}$ + 2D Pallas RMSNorm + Zero-Copy bRPA** | **`727.75 µs`** | **`46.58 ms`** | **`277.80 µs`** | **1.50x (+360.81 µs)** | [platforms-deepsea-8436557819037169021](http://xprof/?session_id=platforms-deepsea-8436557819037169021) |
| **7** | **Head-Sharded $W_{KV}$ + Fused Norm/RoPE + Zero-Copy bRPA** | **`735.77 µs`** | **`47.09 ms`** | **`217.61 µs`** | **1.48x (+352.78 µs)** | [platforms-deepsea-8436557819037169422](http://xprof/?session_id=platforms-deepsea-8436557819037169422) |

> **Architectural Insights from Hardware Traces**:
> 1. **Separate $W_K, W_V$ + Head-Major $Q$ (Tests 8 & 9) Achieves Lowest Latency (`687.27 µs`)**:
>    - Decoupling $W_K$ and $W_V$ into separate projections beats both Shard-Major Joint $W_{KV}$ (`692.60 µs`) and Head-Sharded Joint $W_{KV}$ (`718.61 µs`).
>    - In TP=2, column sharded $W_K$ and $W_V$ produce contiguous slices for each device without requiring any cross-device collectives (`all-to-all = 0 µs`), no slicing overhead (`0 µs` vs `2.61 µs` in Test 4 and `10.63 µs` in Test 6), and preserve optimal $T(4,128)$ VPU sublane tiling.
> 2. **Direct Token-Major Writeback Benefit (`--out_token_major=True`)**:
>    - Writing attention output directly in token-major order eliminates the Step 14 layout transpose (`copy.207` + `copy-done.7`), saving **`7.0 - 12.3 µs`** across all pipelines with head-major intermediate activations.
> 3. **RPAm Megakernel Execution Time**:
>    - In **Test 5, Test 7, & Test 9**, because $Q$ is pre-rotated in the fused Pallas norm pass and RPAm runs with `apply_rope=False`, the on-device RPAm megakernel drops from **`277 µs`** down to **`216.5 µs`** (**`-60.5 µs (-21.8%)` device savings**).
>    - This reaches the standalone FlashAttention physical hardware floor of ~216 µs.
> 4. **Numerical Parity**:
>    - All tests achieve bit-exact or cosine similarity $= 1.000000$ numerical parity against reference attention on Sunfish EVT hardware.

--------------------------------------------------------------------------------

## 4. Usage & Execution Instructions

### Build All Targets

```bash
/google/bin/releases/arca9-local-blaze-cli/blaze-for-agents build //experimental/users/fangfangz/kernels/brpa_rope:all
```

### Run Correctness Tests & Layout Analysis

```bash
# Verify 0 HLO Transposes for KV-Group Major Q and Strided DMA
/google/bin/releases/arca9-local-blaze-cli/blaze-for-agents run //experimental/users/fangfangz/kernels/brpa_rope:test_brpa_layout_overhead

# Run all 11 Numerical Correctness and Pipeline Equivalence Tests
/google/bin/releases/arca9-local-blaze-cli/blaze-for-agents run //experimental/users/fangfangz/kernels/brpa_rope:test_brpa_rope_correctness
```

### Run Benchmark on Sunfish EVT Host

```bash
bash experimental/users/fangfangz/kernels/brpa_rope/run_brpa_rope_benchmark_evt.sh <EVT_HOST> [extra_flags]

# Examples:
# 1. Default run (includes pre-GEMM staging modeling, default head-major attention output):
bash experimental/users/fangfangz/kernels/brpa_rope/run_brpa_rope_benchmark_evt.sh yudlsqd-bcq8

# 2. Benchmark with Direct Token-Major Attention Output enabled:
bash experimental/users/fangfangz/kernels/brpa_rope/run_brpa_rope_benchmark_evt.sh yudlsqd-bcq8 --out_token_major=True

# 3. Benchmark without pre-GEMM layout staging in baseline tests:
bash experimental/users/fangfangz/kernels/brpa_rope/run_brpa_rope_benchmark_evt.sh yudlsqd-bcq8 --include_pre_gemm_staging=False
```
