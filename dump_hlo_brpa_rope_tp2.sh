#!/bin/bash
set -e

source ~/vllm_env/bin/activate
export HF_HOME=~/hf_cache_mount
cd ~/tpu-inference

DUMP_DIR=$HOME/hf_cache_mount/tmp/qwen32b_prefill_brpa_rope_tp2_$(date +%Y%m%d_%H%M%S)
mkdir -p "$DUMP_DIR"
echo "Dumping HLO to $DUMP_DIR with USE_STRIDED_IN_KERNEL_ROPE_RPA=1, USE_BATCHED_RPA_KERNEL=1, USE_FUSED_SWIGLU=1, USE_FUSED_RMSNORM_FP8=1, USE_FUSED_ALL_REDUCE_MATMUL=1..."

sudo rm -rf /tmp/vllm_tmp 2>/dev/null || true
mkdir -p /tmp/vllm_tmp
export TMPDIR=/tmp/vllm_tmp

LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=163840" XLA_FLAGS="--xla_dump_to=$DUMP_DIR --xla_dump_hlo_as_text" EXPECTED_PREFILL_TOKENS=4096 EXPECTED_DECODE_TOKENS=0 USE_STRIDED_IN_KERNEL_ROPE_RPA=1 USE_BATCHED_RPA_KERNEL=1 USE_FUSED_SWIGLU=1 FUSED_SWIGLU_PIPELINE_MODE=grid FUSED_SWIGLU_QUANT_MODE=channelwise_separate_pallas FUSED_SWIGLU_BLOCK_M=2048 FUSED_SWIGLU_BLOCK_N=512 FUSED_SWIGLU_BLOCK_K=5120 USE_FUSED_RMSNORM_FP8=1 FUSED_RMSNORM_BLOCK_M=512 USE_FUSED_ALL_REDUCE_MATMUL=1 FUSED_AR_BLOCK_M=2048 FUSED_AR_BLOCK_N=1024 FUSED_AR_ATTN_BLOCK_K=4096 FUSED_AR_MLP_BLOCK_K=6400 python3 examples/tpu_profiling.py   --model=Qwen/Qwen3-32B   --input-len=4096   --output-len=1   --batch-size=1   --tensor-parallel-size=2   --no-enable-prefix-caching   --additional-config='{"quantization": {"qwix": {"rules": [{"module_path": ".*", "weight_qtype": "float8_e4m3fn", "act_qtype": "float8_e4m3fn"}]}}}'   --kv-cache-dtype=fp8   --max-model-len=8192   --max-num-batched-tokens=8192   --max-num-seqs=256   --gpu-memory-utilization=0.85   --compilation-config='{"cudagraph_capture_sizes": []}'   --hf-overrides='{"num_hidden_layers": 4, "layer_types": ["full_attention", "full_attention", "full_attention", "full_attention"]}'   --load-format dummy

echo "HLO dump complete in $DUMP_DIR"
echo "$DUMP_DIR" > ~/latest_dump_tp2.txt
