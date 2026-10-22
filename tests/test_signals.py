"""Offline tests for the PAPER-ONLY weather signal feed (no network)."""
import datetime as dt
import http.client
import json
import threading
import urllib.parse

from sic.chat_agent import ChatAgent
from sic.llm import MockLLM
from sic.sandbox import LocalSandbox
from sic.server import App, ConversationStore, Server, make_handler
from signals import feed, scorer
from signals import weather_edge as we

UTC = dt.timezone.utc


def test_fee_and_edge():
    assert we.kalshi_fee(0.5) == 0.02 and we.kalshi_fee(0.03) == 0.01
    e = we.edge(0.8, 0.5, 0.52)
    assert e["side"] == "YES" and abs(e["ev"] - (0.8 - 0.5 - 0.02)) < 1e-9 and e["flag"]
    assert not we.edge(0.5, 0.5, 0.5)["flag"]
    assert we.edge(0.5, 0.995, 0.005)["side"] is None  # untradeable extremes are ignored


def test_prob_temp_bins_and_obs():
    # forecast 70 +- tiny: bin 70-71 ~ certain, ">71" ~ none
    assert we.prob_temp([70.0], "between", 70, 71, 0.3) > 0.9
    assert we.prob_temp([70.0], "greater", 71, None, 0.3) < 0.05
    # observed high 75 already, forecast says 70 for the rest: high market must reflect obs
    assert we.prob_temp([70.0], "greater", 73, None, 1.5, "high", obs=75.0) > 0.9
    # observed low 40, rest of day 50: "<42" near certain
    assert we.prob_temp([50.0], "less", None, 42, 1.5, "low", obs=40.0) > 0.9
    # bins over a full partition sum to ~1
    tot = sum(we.prob_temp([70.3, 69.1, 71.8], "between", a, a + 1, 1.5) for a in range(62, 80, 2))  # 0.01 floor clamp adds a little
    assert abs(tot - 1) < 0.08


def test_prob_rain_and_metar_parsing():
    assert we.prob_rain([0, 0, 0.2, 0.5], False) == (2 + 0.5) / 5
    assert we.prob_rain([0, 0], True) == 0.99
    raw = "KNYC 221351Z 00000KT 10SM RA OVC020 13/11 A3000 RMK AO2 P0012 10183 20117 T01330111"
    assert we.metar_precip(raw) == 0.12
    hi, lo = we.metar_6h_extremes(raw)
    assert hi == round(18.3 * 9 / 5 + 32, 1) and lo == round(11.7 * 9 / 5 + 32, 1)


def test_climate_day_uses_standard_time():
    t0, t1 = we.climate_day(dt.date(2026, 7, 1), "America/Chicago")
    assert t0 == dt.datetime(2026, 7, 1, 6, tzinfo=UTC) and t1 - t0 == dt.timedelta(days=1)


def _fixture_build():
    t = dt.datetime(2026, 10, 22, 14, tzinfo=UTC)
    times = [(dt.datetime(2026, 10, 22, 6, tzinfo=UTC) + dt.timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M")
             for h in range(30)]
    ens = {"KXXX": dict(fetched=0, members=2, times=times,
                        temp=[[60 + min(h, 15) for h in range(30)], [61 + min(h, 15) for h in range(30)]],
                        precip=[[0.0] * 30, [0.1] * 30])}
    st = {"KXXX": dict(lat=1, lon=1, site="Testville", tz="America/Chicago")}
    mk = [dict(ticker="KXHIGHX-26OCT22-B75.5", kind="high", icao="KXXX", day="2026-10-22", strike_type="between",
               floor=75, cap=76, close="2026-10-23T05:00:00Z", yes_ask=0.30, no_ask=0.72, yes_bid=0.28),
          dict(ticker="KXRAIN-26OCT22-X", kind="rain", icao="KXXX", day="2026-10-22", strike_type="greater",
               floor=0, cap=None, close="2026-10-23T05:00:00Z", yes_ask=0.5, no_ask=0.52, yes_bid=0.48)]
    return we.build(mk, st, ens, {"KXXX": []}, {"KXXX": {"radar_frac_25km": 0.1}}, t)


def test_build_signals_offline():
    sigs = {s["kind"]: s for s in _fixture_build()}
    assert sigs["high"]["features"]["ens_median_remaining"] == 76
    assert 0.3 < sigs["high"]["p_model"] < 0.9
    assert sigs["rain"]["p_model"] == round((1 + 0.5) / 3, 3)
    assert sigs["rain"]["features"]["radar_frac_25km"] == 0.1


def test_scorer_brier_and_paper_pnl():
    rows = [dict(t="2026-10-22T10:00", ticker="A", kind="rain", p=0.9, mid=0.5,
                 edge=dict(side="YES", ev=0.3, price=0.5, flag=True)),
            dict(t="2026-10-22T11:00", ticker="A", kind="rain", p=0.8, mid=0.6,
                 edge=dict(side="YES", ev=0.1, price=0.6, flag=True)),
            dict(t="2026-10-22T10:00", ticker="B", kind="high", p=0.2, mid=0.5,
                 edge=dict(side="NO", ev=0.2, price=0.5, flag=True)),
            dict(t="2026-10-22T10:00", ticker="C", kind="high", p=0.2, mid=0.5, edge=dict(flag=False))]
    s = scorer.score(rows, {"A": "yes", "B": "yes"})   # C unsettled -> ignored
    a = s["by_kind"]["all"]
    assert a["tickers"] == 2 and a["paper_trades"] == 2 and a["wins"] == 1
    assert abs(a["paper_pnl_per_contract_sum"] - ((1 - 0.5 - 0.02) + (0 - 0.5 - 0.02))) < 1e-9
    assert abs(s["by_kind"]["rain"]["brier_model"] - (0.01 + 0.04) / 2) < 1e-9


def _write_feed(d):
    sig = dict(ticker="KXRAIN-26OCT22-NYC", kind="rain", city="New York/Central Park", p_model=0.7,
               yes_ask=0.4, no_ask=0.62, edge=dict(side="YES", ev=0.28, price=0.4, flag=True),
               features=dict(hours_left=10))
    json.dump(dict(generated="2026-10-22T13:00:00+00:00", n_signals=1, n_flags=1, signals=[sig]),
              open(d / "signals.json", "w"))


def test_chat_tool_and_prompt_summary(tmp_path):
    _write_feed(tmp_path)
    llm = MockLLM(['<tool>{"name": "get_trade_signals", "args": {"kind": "rain"}}</tool>', "Paper only: NYC rain."])
    events = []
    out = ChatAgent(llm, LocalSandbox(), web=False, signals_dir=str(tmp_path)).respond(
        "any kalshi rain edge today?", emit=events.append)
    first = llm.prompts[0]
    assert "get_trade_signals" in first[0]["content"]                 # tool advertised
    assert "KXRAIN-26OCT22-NYC" in first[-1]["content"]                # summary injected
    assert "KXRAIN-26OCT22-NYC" in llm.prompts[1][-1]["content"] and "untrusted" in llm.prompts[1][-1]["content"]
    assert out["answer"].startswith("Paper only")
    assert not feed.wants_signals("write a python sort")


def test_signals_page_requires_auth(tmp_path):
    _write_feed(tmp_path)
    app = App(lambda m: None, "pw-123", store=ConversationStore(str(tmp_path / "c")), signals_dir=str(tmp_path))
    srv = Server(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]

    def req(method, path, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        c.request(method, path, body=body, headers=headers or {})
        r = c.getresponse()
        return r, r.read().decode()
    try:
        for p in ("/signals", "/api/signals"):
            assert req("GET", p)[0].status == 303
        r, _ = req("POST", "/login", urllib.parse.urlencode({"password": "pw-123"}),
                   {"Content-Type": "application/x-www-form-urlencoded"})
        ck = r.getheader("Set-Cookie").split(";")[0]
        r, page = req("GET", "/signals", headers={"Cookie": ck})
        assert r.status == 200 and "PAPER ONLY" in page and "viewport" in page
        r, body = req("GET", "/api/signals?kind=rain&flagged=1", headers={"Cookie": ck})
        d = json.loads(body)
        assert d["paper_only"] and d["live_trading"] is False and d["signals"][0]["ticker"] == "KXRAIN-26OCT22-NYC"
    finally:
        srv.shutdown()
