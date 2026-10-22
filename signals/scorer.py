"""PAPER-ONLY scorer: compares logged signals to settled Kalshi results (public GET /markets/{ticker}).

For every settled ticker it reports
  * Brier score of the model probability vs Brier of the market mid, averaged over all snapshots,
  * paper P&L of the FIRST flagged snapshot per ticker (1 contract at the logged ask, minus taker fee).
Writes data/scores.json and data/settled.json (result cache).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from collections import defaultdict

from . import weather_edge as we


def score(log_rows: list[dict], results: dict) -> dict:
    by = defaultdict(list)
    for r in log_rows:
        if results.get(r["ticker"]) in ("yes", "no"):
            by[r["ticker"]].append(r)
    agg = defaultdict(lambda: dict(tickers=0, snaps=0, brier_model=0.0, brier_mkt=0.0, mkt_snaps=0,
                                   trades=0, wins=0, pnl=0.0))
    trades = []
    for tk, rows in by.items():
        y = 1.0 if results[tk] == "yes" else 0.0
        for kind in (rows[0]["kind"], "all"):
            a = agg[kind]
            a["tickers"] += 1
            for r in rows:
                a["snaps"] += 1
                a["brier_model"] += (r["p"] - y) ** 2
                if r.get("mid") is not None:
                    a["mkt_snaps"] += 1
                    a["brier_mkt"] += (r["mid"] - y) ** 2
        first = next((r for r in sorted(rows, key=lambda r: r["t"]) if r["edge"].get("flag")), None)
        if first:
            e = first["edge"]
            win = (e["side"] == "YES") == (y == 1.0)
            pnl = round((1.0 if win else 0.0) - e["price"] - we.kalshi_fee(e["price"]), 4)
            trades.append(dict(ticker=tk, t=first["t"], side=e["side"], price=e["price"], p=first["p"],
                               result=results[tk], pnl=pnl))
            for kind in (rows[0]["kind"], "all"):
                agg[kind]["trades"] += 1
                agg[kind]["wins"] += win
                agg[kind]["pnl"] = round(agg[kind]["pnl"] + pnl, 4)
    out = {}
    for k, a in agg.items():
        out[k] = dict(tickers=a["tickers"], snaps=a["snaps"],
                      brier_model=round(a["brier_model"] / a["snaps"], 4) if a["snaps"] else None,
                      brier_market=round(a["brier_mkt"] / a["mkt_snaps"], 4) if a["mkt_snaps"] else None,
                      paper_trades=a["trades"], wins=a["wins"], paper_pnl_per_contract_sum=a["pnl"])
    return dict(by_kind=out, trades=trades[-200:])


def run_once() -> dict:
    d = we.DATA
    logf = os.path.join(d, "signals_log.jsonl")
    if not os.path.exists(logf):
        return {}
    rows = [json.loads(x) for x in open(logf) if x.strip()]
    cache_f = os.path.join(d, "settled.json")
    results = json.load(open(cache_f)) if os.path.exists(cache_f) else {}
    today = dt.datetime.now(we.UTC).date()
    pending = sorted({r["ticker"] for r in rows if r["ticker"] not in results})
    for tk in pending:
        try:
            day = dt.datetime.strptime(tk.split("-")[1], "%y%b%d").date()
        except (IndexError, ValueError):
            continue
        if day >= today:
            continue
        try:
            m = we.S.get(f"{we.KALSHI}/markets/{tk}", timeout=20).json().get("market", {})
        except Exception as e:  # noqa: BLE001
            print("settle", tk, type(e).__name__, file=sys.stderr)
            continue
        if m.get("result") in ("yes", "no"):
            results[tk] = m["result"]
        time.sleep(0.1)
    json.dump(results, open(cache_f, "w"))
    sc = score(rows, results)
    sc.update(generated=dt.datetime.now(we.UTC).isoformat(), settled_tickers=len(results),
              logged_tickers=len({r["ticker"] for r in rows}), paper_only=True,
              how="Brier: lower is better; model must beat market to claim edge. P&L: first flagged "
                  "snapshot per ticker, 1 contract at logged ask minus taker fee.")
    tmp = os.path.join(d, "scores.json.tmp")
    json.dump(sc, open(tmp, "w"), indent=1)
    os.replace(tmp, os.path.join(d, "scores.json"))
    return sc


if __name__ == "__main__":
    s = run_once()
    print(json.dumps(s.get("by_kind", {}), indent=1))
