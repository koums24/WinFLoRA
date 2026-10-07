#!/bin/bash
# Progress report for the remote training run.
cd "$(dirname "$0")/.."      # Src/
date
grep -aoE "Client [0-9]+ (Starting Training|Results|Failed)[^{]{0,40}" train_agnews.log | tail -12
tail -c 250 train_agnews.log | tr '\r' '\n' | tail -2
echo
grep -aoE "Client [0-9]+ Results: \{'eval_loss': [0-9.]+, 'eval_accuracy': [0-9.]+" train_agnews.log
echo "finished adapters: $(ls noisy_tiny_agnews/artifacts/*/lora.pt 2>/dev/null | wc -l)/10"
pgrep -f tiny_pca_greedy >/dev/null && echo RUNNING || echo STOPPED
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
df -h / | tail -1
