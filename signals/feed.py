"""Read-only helpers that turn signals.json / scores.json into tool output and a short prompt summary."""
from __future__ import annotations

import json
import os

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
KEYWORDS = ("kalshi", "trade", "trading", "signal", "weather", "rain", "temperature", "temp ", "edge", "market")


def _load(d, name):
    try:
        with open(os.path.join(d, name)) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def get_trade_signals(data_dir: str = DEFAULT_DIR, kind: str | None = None, city: str | None = None,
                      flagged_only: bool = False, limit: int = 15) -> dict:
    doc = _load(data_dir, "signals.json") or {}
    sc = _load(data_dir, "scores.json") or {}
    sig = doc.get("signals", [])
    if kind:
        sig = [s for s in sig if s.get("kind") == kind]
    if city:
        c = city.lower()
        sig = [s for s in sig if c in (s.get("city") or "").lower() or c in s["ticker"].lower()]
    if flagged_only:
        sig = [s for s in sig if s.get("edge", {}).get("flag")]
    slim = [dict(ticker=s["ticker"], kind=s["kind"], city=s.get("city"), p_model=s["p_model"],
                 yes_ask=s["yes_ask"], no_ask=s["no_ask"], side=s["edge"]["side"], ev=s["edge"]["ev"],
                 flag=s["edge"]["flag"], hours_left=s["features"].get("hours_left")) for s in sig[:limit]]
    return dict(paper_only=True, live_trading=False, generated=doc.get("generated"),
                n_signals=doc.get("n_signals", 0), n_flags=doc.get("n_flags", 0), signals=slim,
                scores=sc.get("by_kind", {}), settled_tickers=sc.get("settled_tickers", 0),
                note="PAPER ONLY. Probabilities are uncalibrated until scores show the model beating the market.")


def wants_signals(message: str) -> bool:
    m = (message or "").lower()
    return any(k in m for k in KEYWORDS)


def summary(data_dir: str = DEFAULT_DIR, top: int = 5) -> str:
    r = get_trade_signals(data_dir, flagged_only=True, limit=top)
    if not r["generated"]:
        return ""
    lines = [f"[weather signal feed, PAPER ONLY, generated {r['generated']}: {r['n_signals']} markets, "
             f"{r['n_flags']} paper flags]"]
    for s in r["signals"]:
        lines.append(f"- {s['ticker']}: model p(yes)={s['p_model']}, yes_ask={s['yes_ask']}, no_ask={s['no_ask']}"
                     f" -> paper {s['side']} ev={s['ev']}")
    a = r["scores"].get("all")
    if a:
        lines.append(f"score so far: {a['tickers']} settled tickers, Brier model={a['brier_model']} vs "
                     f"market={a['brier_market']}, paper P&L={a['paper_pnl_per_contract_sum']} per contract")
    else:
        lines.append("score so far: nothing settled yet, so no evidence of edge.")
    lines.append("Never claim live trading; live trading is OFF.")
    return "\n".join(lines)
