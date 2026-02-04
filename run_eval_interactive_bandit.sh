#!/usr/bin/env bash
# Evaluate the interactive bandit model (after training with train_interactive.py).
# Usage: ./run_eval_interactive_bandit.sh
# Or:    python evals/eval_interactive_bandit.py --dim 5 --H 100 --n_eval 100 --model_path models/interactive_bandit.pt

python evals/eval_interactive_bandit.py \
  --env bandit \
  --dim 5 \
  --H 100 \
  --n_eval 100 \
  --var 0.3 \
  --model_path models/interactive_bandit.pt \
  --save figs/eval_interactive_bandit
