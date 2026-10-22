# signals/ — PAPER-ONLY Kalshi weather signal feed

No orders, no keys, no money. Public keyless data only:
Kalshi public market data (GET), Open-Meteo ensemble (NOAA GEFS 0.25 + ECMWF IFS ENS 0.25, the models
gods-eye-view animates), NOAA NBM via Open-Meteo, AWC METAR, NOAA nowCOAST MRMS radar + lightning
(same WMS layers as gods-eye-view `server/providers/weather.js`).

Markets: KXRAIN plus the US KXHIGH*/KXLOWT* daily series; settlement station parsed from each market's
`CLIxxx` rule text (same approach as /workspace/rain/krain_watch.py).

Model
* high/low: per ensemble member, extreme over the remaining climate-day hours (LST midnight-midnight),
  shifted so the ensemble median matches NBM; combined with the observed extreme so far
  (METAR temps + 6-h max/min groups); normal kernel for rounding / forecast error.
* rain: locked (0.99) once a METAR P-group >= 0.01 in falls inside the climate day; otherwise fraction of
  ensemble members with >= 0.01 in over remaining hours. Radar/lightning fractions are context features only.
* edge: EV per contract = p - ask - taker fee; paper flag if EV >= 0.10.

Run: `nohup setsid signals/loop.sh &` (every 15 min; `WEATHER_EDGE_INTERVAL` seconds).
Files in `signals/data/` (gitignored): `signals.json`, `signals_log.jsonl`, `scores.json`, `settled.json`,
`loop.out`, `loop.err`, `loop.pid`. Stop: `kill $(cat signals/data/loop.pid)`.

Scoring (`python3 -m signals.scorer`): for each ticker whose date has passed, fetch the public result,
then Brier(model) vs Brier(market mid) over every logged snapshot, and paper P&L of the first flagged
snapshot per ticker (1 contract at logged ask minus fee). The model has an edge only if Brier(model) <
Brier(market) and paper P&L > 0 over a few hundred settled tickers.

Chat: `get_trade_signals` tool in `sic/chat_agent.py`; a short summary is injected when a message
mentions kalshi/trade/weather/rain/etc. Phone: `/signals` (login required) on the chat server.
