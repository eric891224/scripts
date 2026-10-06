#!/usr/bin/env bash
set -euo pipefail

# Source your environment manually before running this script.
for setting in VLLM_BIN MODEL SERVED_MODEL_NAME PORT TENSOR_PARALLEL_SIZE; do
    if [[ -z "${!setting:-}" ]]; then
        echo "Missing ${setting}; set it before serving." >&2
        exit 2
    fi
done

exec "$VLLM_BIN" serve "$MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.85 \
    --language-model-only \
    --generation-config vllm \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder