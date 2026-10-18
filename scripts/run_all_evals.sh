#!/usr/bin/env bash
# Runs every condition in order (each resumable), committing after each one.
set -uo pipefail
cd "$(dirname "$0")/.."
for c in a b learn c; do
  python3 -m evals.run_eval --cond "$c" || exit 1
  python3 -m evals.run_eval --summary >/dev/null
  git add results evals/splits.json store/experience_*.jsonl data/finetune docs/results.svg 2>/dev/null
  git commit -q -m "eval: condition $c results" || true
done
python3 -m evals.run_eval --summary
