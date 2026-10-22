#!/usr/bin/env bash
# up: start GPU Studio + vLLM, and make sure the idle watchdog runs.  down: stop now.  status.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
case "${1:-status}" in
  up)
    pgrep -f "lightning_llm.py watch" >/dev/null || nohup python3 deploy/lightning_llm.py watch >> logs/lightning_watch.log 2>&1 &
    python3 deploy/lightning_llm.py up 2>&1 | tee -a logs/lightning_up.log ;;
  down) python3 deploy/lightning_llm.py down ;;
  status) python3 deploy/lightning_llm.py status ;;
  *) echo "usage: $0 up|down|status"; exit 2 ;;
esac
