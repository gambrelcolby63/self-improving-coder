#!/usr/bin/env bash
# PAPER-ONLY signal loop: build signals every 15 min, score every cycle. No orders, no keys.
cd "$(dirname "$0")/.." || exit 1
mkdir -p signals/data
echo $$ > signals/data/loop.pid
while true; do
  timeout 600 python3 -m signals.weather_edge >> signals/data/loop.out 2>> signals/data/loop.err
  timeout 300 python3 -m signals.scorer > /dev/null 2>> signals/data/loop.err
  sleep "${WEATHER_EDGE_INTERVAL:-900}"
done
