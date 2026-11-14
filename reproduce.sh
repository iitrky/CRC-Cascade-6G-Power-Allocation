#!/usr/bin/env bash
# Regenerates every table and figure of the paper (about 40 min on 2 CPU cores).
set -euo pipefail
python -m tests.test_autodiff
python -m scripts.train
python -m scripts.evaluate
python -m scripts.evaluate_cascade
python -m scripts.cascade_online
python -m scripts.make_figures
echo "Done. Numbers: results/results.json, results/results_cascade.json; figures: results/figs/"
