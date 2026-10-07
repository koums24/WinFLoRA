#!/bin/bash
# Waits for the training run to exit, then runs every attack on the 10 real adapters.
cd "$(dirname "$0")/.."      # Src/ (attack_sim.py imports rpca/utils from there)
. ../.venv/bin/activate
while pgrep -f tiny_pca_greedy_agnews >/dev/null; do sleep 60; done
echo "training exited: $(date)" > pipeline.log
for e in repo paper; do
  python attack_sim.py --source artifacts --num_clients 10 --estimator $e --out attack_results >> pipeline.log 2>&1
done
echo "DONE: $(date)" >> pipeline.log
