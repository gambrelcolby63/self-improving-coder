"""PAPER-ONLY Kalshi weather signal job.

Data (all keyless, public):
  * Kalshi public market data   api.elections.kalshi.com/trade-api/v2 (GET only, no auth)
  * Open-Meteo ensemble API     GEFS 0.25 (31 members) + ECMWF IFS 0.25 ENS (51 members)
                                -- the same NOAA GFS / ECMWF IFS models gods-eye-view animates
  * AWC METAR                   observed temp / precip so far in the climate day
  * NOAA nowCOAST WMS           MRMS radar reflectivity + lightning density near the station
                                (same layers as gods-eye-view server/providers/weather.js)

Output: data/signals.json (latest) and data/signals_log.jsonl (append).
No orders. No keys. No money moves.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import math
import os
import re
import sys
import time
from zoneinfo import ZoneInfo

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("WEATHER_EDGE_DATA", os.path.join(HERE, "data"))
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
ENS = "https://ensemble-api.open-meteo.com/v1/ensemble"
AWC = "https://aviationweather.gov/api/data"
NOWCOAST = "https://nowcoast.noaa.gov/geoserver/observations/"
UTC = dt.timezone.utc
UA = {"User-Agent": "weather-edge-paper (personal research; no trading)"}
ENS_TTL = 2 * 3600        # models update every 6 h; keep Open-Meteo usage low
RAIN_MEMBER_IN = 0.03     # per-member rain threshold (drizzle correction)
MARKET_CONFIDENT = 0.93   # do not fade a market this confident unless observations contradict it
EDGE_MIN = 0.10           # paper flag threshold (EV per contract after fee)
# US daily temperature/rain series (from /workspace/settle/series.json; intl ones have no CLI station and are skipped)
SERIES = ["KXRAIN", "KXHIGHAUS", "KXHIGHCHI", "KXHIGHDEN", "KXHIGHLAX", "KXHIGHMIA", "KXHIGHNY", "KXHIGHPHIL",
          "KXHIGHTATL", "KXHIGHTBOS", "KXHIGHTDAL", "KXHIGHTDC", "KXHIGHTHOU", "KXHIGHTLV", "KXHIGHTMIN",
          "KXHIGHTNOLA", "KXHIGHTOKC", "KXHIGHTPHX", "KXHIGHTSATX", "KXHIGHTSEA", "KXHIGHTSFO", "KXHIGHTEWR",
          "KXHIGHTTTN", "KXHIGHTKSAN", "KXHIGHTSDF", "KXLOWTATL", "KXLOWTAUS", "KXLOWTBOS", "KXLOWTCHI",
          "KXLOWTDAL", "KXLOWTDC", "KXLOWTDEN", "KXLOWTEWR", "KXLOWTHOU", "KXLOWTLAX", "KXLOWTLV", "KXLOWTMIA",
          "KXLOWTMIN", "KXLOWTNOLA", "KXLOWTNYC", "KXLOWTOKC", "KXLOWTPHIL", "KXLOWTPHX", "KXLOWTSATX",
          "KXLOWTSEA", "KXLOWTSFO", "KXLOWTTTN", "KXLOWTKSAN", "KXLOWTSDF", "KXLOWTSAN"]
CLI2ICAO = {"NYC": "KNYC"}  # default is 'K'+code

S = requests.Session()
S.headers.update(UA)


def now() -> dt.datetime:
    return dt.datetime.now(UTC)


# ---------------------------------------------------------------- pure math (tested)
def phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def kalshi_fee(p: float) -> float:
    """Taker fee per contract, ceil(0.07*p*(1-p)) to the cent."""
    return math.ceil(round(0.07 * p * (1 - p) * 100, 6)) / 100


def prob_temp(values: list[float], strike_type: str, floor, cap, sigma: float,
              kind: str = "high", obs=None, sigma_obs: float = 0.6) -> float:
    """P(reported integer °F satisfies the contract).

    values: per-member forecast extreme over the REMAINING hours of the climate day (bias-corrected).
    obs: observed extreme so far (None if no obs). Day extreme X = max(obs, F) for highs, min(obs, F) for lows,
    with F ~ N(member, sigma) and obs ~ N(obs, sigma_obs) (METAR vs CLI rounding). Averaged over members.
    """
    if not values and obs is None:
        return float("nan")
    values = values or [None]

    def cdf(x, v):  # P(X <= x)
        if kind == "high":
            a = phi((x - obs) / sigma_obs) if obs is not None else 1.0
            b = phi((x - v) / sigma) if v is not None else 1.0
            return a * b
        a = 1 - phi((x - obs) / sigma_obs) if obs is not None else 1.0
        b = 1 - phi((x - v) / sigma) if v is not None else 1.0
        return 1 - a * b

    ps = []
    for v in values:
        if strike_type == "greater":      # > floor  ->  X >= floor+1
            p = 1 - cdf(floor + 0.5, v)
        elif strike_type == "less":       # < cap    ->  X <= cap-1
            p = cdf(cap - 0.5, v)
        else:                             # between, inclusive
            p = cdf(cap + 0.5, v) - cdf(floor - 0.5, v)
        ps.append(p)
    return min(0.99, max(0.01, sum(ps) / len(ps)))


def anchor_shift(member_vals: list[float], anchor) -> float:
    """Shift that moves the ensemble median onto the NBM anchor (NBM is the bias-corrected NWS blend;
    raw 0.25-degree ensembles run cold at the daily peak). Ensemble keeps supplying the spread."""
    if anchor is None or not member_vals:
        return 0.0
    med = sorted(member_vals)[len(member_vals) // 2]
    return max(-8.0, min(8.0, anchor - med))


def prob_rain(member_totals: list[float], locked: bool) -> float:
    if locked:
        return 0.99
    if not member_totals:
        return float("nan")
    # drizzle correction: raw ensembles over-forecast trace amounts, so a member only counts as "rain"
    # at >= RAIN_MEMBER_IN (settlement needs a measurable >= 0.01 in at the gauge)
    k = sum(1 for t in member_totals if t >= RAIN_MEMBER_IN)
    return min(0.98, max(0.02, (k + 0.5) / (len(member_totals) + 1)))


def obs_decided(kind: str, strike_type: str, floor, cap, obs_extreme=None, rain_locked=False):
    """'YES'/'NO' if observations so far already settle the contract (1 F margin for METAR vs CLI), else None."""
    if kind == "rain":
        return "YES" if rain_locked else None
    if obs_extreme is None:
        return None
    r = round(obs_extreme)
    if kind == "high":           # the day's max can only go up
        if strike_type == "greater" and r >= floor + 2:
            return "YES"
        if strike_type == "less" and r >= cap + 1:
            return "NO"
        if strike_type == "between" and r >= cap + 2:
            return "NO"
    else:                        # the day's min can only go down
        if strike_type == "less" and r <= cap - 2:
            return "YES"
        if strike_type == "greater" and r <= floor - 1:
            return "NO"
        if strike_type == "between" and r <= floor - 2:
            return "NO"
    return None


def gate(e: dict, decided, yes_ask, no_ask) -> dict:
    """Locked-market filter applied to a raw edge."""
    e = dict(e)
    conf = max(yes_ask or 0, no_ask or 0) >= MARKET_CONFIDENT
    if decided:
        # outcome already set by observations: only a paper flag if we're on that side AND the market disagrees
        ok = e["side"] == decided and (yes_ask if decided == "YES" else no_ask or 1) <= 0.90
        e["gate"] = "obs_decided_" + decided + ("_market_disagrees" if ok else "")
        e["flag"] = bool(e["flag"] and ok)
    elif conf:
        e["gate"] = "market_confident"
        e["flag"] = False
    return e


def edge(p: float, yes_ask, no_ask) -> dict:
    out = {"side": None, "ev": 0.0, "price": None}
    for side, prob, ask in (("YES", p, yes_ask), ("NO", 1 - p, no_ask)):
        if ask is None or not (0.02 <= ask <= 0.98) or p != p:
            continue
        ev = prob - ask - kalshi_fee(ask)
        if ev > out["ev"]:
            out = {"side": side, "ev": round(ev, 4), "price": ask}
    out["flag"] = out["ev"] >= EDGE_MIN
    return out


def climate_day(day: dt.date, tz: str) -> tuple[dt.datetime, dt.datetime]:
    """NWS climate day = local STANDARD time midnight-to-midnight."""
    off = dt.datetime(day.year, 1, 15, tzinfo=ZoneInfo(tz)).utcoffset()
    t0 = dt.datetime(day.year, day.month, day.day, tzinfo=UTC) - off
    return t0, t0 + dt.timedelta(days=1)


def metar_6h_extremes(raw: str):
    """6-hourly max/min groups 1snTTT / 2snTTT in RMK (tenths degC) -> (max_f, min_f) or None."""
    body = raw.split(" RMK ", 1)[1] if " RMK " in raw else ""
    out = []
    for lead in ("1", "2"):
        g = re.search(r"\b" + lead + r"([01])(\d{3})\b", body)
        if g:
            c = int(g.group(2)) / 10 * (-1 if g.group(1) == "1" else 1)
            out.append(round(c * 9 / 5 + 32, 1))
        else:
            out.append(None)
    return tuple(out)


def metar_precip(raw: str) -> float:
    body = raw.split(" RMK ", 1)[1] if " RMK " in raw else ""
    return max([int(g) / 100 for g in re.findall(r"\bP(\d{4})\b", body)] or [0.0])


# ---------------------------------------------------------------- fetchers
def fdollar(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def kalshi_markets() -> list[dict]:
    out = []
    for s in SERIES:
        try:
            for attempt in range(3):
                r = S.get(f"{KALSHI}/events", params={"series_ticker": s, "status": "open",
                                                       "with_nested_markets": "true", "limit": 20}, timeout=20)
                if r.status_code != 429:
                    break
                time.sleep(2 * (attempt + 1))
            j = r.json()
        except Exception as e:  # noqa: BLE001
            print("kalshi", s, type(e).__name__, file=sys.stderr)
            continue
        for e in j.get("events", []):
            for m in e.get("markets", []):
                g = re.search(r"\bCLI([A-Z]{3})\b", m.get("rules_primary", ""))
                if not g:
                    continue
                try:
                    day = dt.datetime.strptime(m["ticker"].split("-")[1], "%y%b%d").date()
                except (IndexError, ValueError):
                    continue
                kind = "rain" if s == "KXRAIN" else ("high" if s.startswith("KXHIGH") else "low")
                out.append(dict(ticker=m["ticker"], series=s, kind=kind, cli=g.group(1),
                                icao=CLI2ICAO.get(g.group(1), "K" + g.group(1)), day=day.isoformat(),
                                strike_type=m.get("strike_type"), floor=m.get("floor_strike"),
                                cap=m.get("cap_strike"), close=m.get("close_time"),
                                yes_ask=fdollar(m.get("yes_ask_dollars")), no_ask=fdollar(m.get("no_ask_dollars")),
                                yes_bid=fdollar(m.get("yes_bid_dollars")), volume=m.get("volume_fp") or m.get("volume"),
                                title=(m.get("title") or "")[:120]))
        time.sleep(0.25)
    return out


def stations(icaos: list[str]) -> dict:
    cache_f = os.path.join(DATA, "stations.json")
    cache = json.load(open(cache_f)) if os.path.exists(cache_f) else {}
    need = [i for i in icaos if i not in cache]
    if need:
        for x in S.get(f"{AWC}/stationinfo", params={"ids": ",".join(need), "format": "json"}, timeout=30).json():
            cache[x["icaoId"]] = dict(lat=x["lat"], lon=x["lon"], site=x.get("site"), state=x.get("state"),
                                      tz=None)
        # timezone via Open-Meteo (keyless) for any new station
        for i in need:
            if i in cache and not cache[i]["tz"]:
                try:
                    r = S.get("https://api.open-meteo.com/v1/forecast", params={
                        "latitude": cache[i]["lat"], "longitude": cache[i]["lon"], "timezone": "auto",
                        "forecast_days": 1, "daily": "temperature_2m_max"}, timeout=20).json()
                    cache[i]["tz"] = r.get("timezone")
                except Exception:  # noqa: BLE001
                    pass
        json.dump(cache, open(cache_f, "w"), indent=1)
    return cache


def ensemble(st: dict) -> dict:
    """Hourly ensemble members per station, cached ENS_TTL. Returns icao -> {times, temp:[[..]], precip:[[..]]}."""
    cache_f = os.path.join(DATA, "ens_cache.json")
    cache = json.load(open(cache_f)) if os.path.exists(cache_f) else {}
    stale = [i for i in st if time.time() - cache.get(i, {}).get("fetched", 0) > ENS_TTL]
    for k in range(0, len(stale), 10):
        chunk = stale[k:k + 10]
        try:
            r = S.get(ENS, params={
                "latitude": ",".join(str(st[i]["lat"]) for i in chunk),
                "longitude": ",".join(str(st[i]["lon"]) for i in chunk),
                "hourly": "temperature_2m,precipitation", "models": "gfs025,ecmwf_ifs025",
                "temperature_unit": "fahrenheit", "precipitation_unit": "inch",
                "timezone": "GMT", "past_days": 1, "forecast_days": 3}, timeout=60)
            r.raise_for_status()
            js = r.json()
        except Exception as e:  # noqa: BLE001
            print("ensemble", type(e).__name__, str(e)[:120], file=sys.stderr)
            continue
        js = js if isinstance(js, list) else [js]
        for i, j in zip(chunk, js):
            h = j["hourly"]
            temp = [v for key, v in h.items() if key.startswith("temperature_2m")]
            pr = [v for key, v in h.items() if key.startswith("precipitation")]
            cache[i] = dict(fetched=time.time(), times=h["time"], temp=temp, precip=pr,
                            members=len(temp))
        time.sleep(1)
    json.dump(cache, open(cache_f, "w"))
    return cache


def nbm(st: dict) -> dict:
    """NOAA National Blend of Models hourly temp via Open-Meteo (keyless), cached 1 h."""
    cache_f = os.path.join(DATA, "nbm_cache.json")
    cache = json.load(open(cache_f)) if os.path.exists(cache_f) else {}
    stale = [i for i in st if time.time() - cache.get(i, {}).get("fetched", 0) > 3600]
    for k in range(0, len(stale), 20):
        chunk = stale[k:k + 20]
        try:
            r = S.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude": ",".join(str(st[i]["lat"]) for i in chunk),
                "longitude": ",".join(str(st[i]["lon"]) for i in chunk),
                "hourly": "temperature_2m", "models": "ncep_nbm_conus", "temperature_unit": "fahrenheit",
                "timezone": "GMT", "past_days": 1, "forecast_days": 3}, timeout=60)
            r.raise_for_status()
            js = r.json()
        except Exception as e:  # noqa: BLE001
            print("nbm", type(e).__name__, str(e)[:120], file=sys.stderr)
            continue
        js = js if isinstance(js, list) else [js]
        for i, j in zip(chunk, js):
            cache[i] = dict(fetched=time.time(), times=j["hourly"]["time"], temp=j["hourly"]["temperature_2m"])
    json.dump(cache, open(cache_f, "w"))
    return cache


def metars(icaos: list[str]) -> dict:
    out = {i: [] for i in icaos}
    for k in range(0, len(icaos), 40):
        try:
            for x in S.get(f"{AWC}/metar", params={"ids": ",".join(icaos[k:k + 40]), "format": "json",
                                                     "hours": 36}, timeout=30).json():
                if x.get("icaoId") in out:
                    out[x["icaoId"]].append(dict(t=x["obsTime"], temp_f=None if x.get("temp") is None
                                                 else x["temp"] * 9 / 5 + 32, precip=metar_precip(x.get("rawOb") or ""),
                                                 ext6=metar_6h_extremes(x.get("rawOb") or "")))
        except Exception as e:  # noqa: BLE001
            print("metar", type(e).__name__, file=sys.stderr)
    return out


def wms_fraction(service: str, layer: str, lat: float, lon: float, half_deg: float) -> float | None:
    """Fraction of non-transparent pixels in a small WMS tile around the point (radar echo / lightning present)."""
    try:
        from PIL import Image
        r = S.get(f"{NOWCOAST}{service}/ows", params={
            "service": "WMS", "version": "1.1.1", "request": "GetMap", "layers": layer, "styles": "",
            "srs": "EPSG:4326", "bbox": f"{lon - half_deg},{lat - half_deg},{lon + half_deg},{lat + half_deg}",
            "width": 48, "height": 48, "format": "image/png", "transparent": "true"}, timeout=20)
        if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image/png"):
            return None
        im = Image.open(io.BytesIO(r.content)).convert("RGBA")
        px = im.get_flattened_data() if hasattr(im, "get_flattened_data") else im.getdata()
        a = [p[3] for p in px]
        return round(sum(1 for v in a if v > 40) / len(a), 3)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- signal build
def build(markets, st, ens, obs, wx, t=None, nb=None) -> list[dict]:
    t = t or now()
    sigs = []
    for m in markets:
        s = st.get(m["icao"])
        if not s or not s.get("tz") or m["icao"] not in ens:
            continue
        t0, t1 = climate_day(dt.date.fromisoformat(m["day"]), s["tz"])
        if t >= t1:
            continue
        e = ens[m["icao"]]
        times = [dt.datetime.fromisoformat(x).replace(tzinfo=UTC) for x in e["times"]]
        # remaining hours of the climate day (hourly values are end-of-hour; include hours ending after now)
        idx = [k for k, x in enumerate(times) if t0 < x <= t1 and x > t]
        o = [x for x in obs.get(m["icao"], []) if t0.timestamp() <= x["t"] < t1.timestamp()]
        ot = [x["temp_f"] for x in o if x["temp_f"] is not None]
        # 6-h max/min groups catch extremes between hourly obs; only use ones whose 6-h window is inside the day
        for x in o:
            if x.get("ext6") and x["t"] - 6 * 3600 >= t0.timestamp() - 600:  # METAR :53 windows
                hi, lo = x["ext6"]
                ot += [v for v in ((hi if m["kind"] == "high" else lo),) if v is not None]
        hours_left = max(0.0, (t1 - t).total_seconds() / 3600)
        feat = dict(hours_left=round(hours_left, 1), obs_n=len(o), members=e["members"],
                    ens_age_min=round((time.time() - e["fetched"]) / 60))
        if m["kind"] in ("high", "low"):
            agg = max if m["kind"] == "high" else min
            vals = []
            for mem in e["temp"]:
                fut = [mem[k] for k in idx if mem[k] is not None]
                if fut:
                    vals.append(agg(fut))
            nbm_x = None
            n = (nb or {}).get(m["icao"])
            if n:
                nv = [v for x, v in zip(n["times"], n["temp"]) if v is not None
                      and t0 < dt.datetime.fromisoformat(x).replace(tzinfo=UTC) <= t1
                      and dt.datetime.fromisoformat(x).replace(tzinfo=UTC) > t]
                nbm_x = agg(nv) if nv else None
            bias = anchor_shift(vals, nbm_x)
            vals = [v + bias for v in vals]
            sigma = 0.8 + min(1.2, hours_left / 16)   # extra kernel on top of ensemble spread
            ox = agg(ot) if ot else None
            p = prob_temp(vals, m["strike_type"], m["floor"], m["cap"], sigma, m["kind"], ox)
            decided = obs_decided(m["kind"], m["strike_type"], m["floor"], m["cap"], ox)
            feat.update(obs_extreme=round(ox, 1) if ox is not None else None, bias_f=round(bias, 2), nbm_remaining=nbm_x,
                        ens_median_remaining=round(sorted(vals)[len(vals) // 2], 1) if vals else None,
                        sigma=round(sigma, 2))
        else:
            locked = any(x["precip"] >= 0.01 for x in o)
            tot = [sum(mem[k] or 0 for k in idx) for mem in e["precip"]]
            p = prob_rain(tot, locked)
            decided = obs_decided("rain", None, None, None, rain_locked=locked)
            feat.update(locked=locked, obs_precip=round(sum(x["precip"] for x in o), 2),
                        **wx.get(m["icao"], {}))
        if p != p:
            continue
        mid = None
        if m["yes_bid"] is not None and m["yes_ask"] is not None:
            mid = round((m["yes_bid"] + m["yes_ask"]) / 2, 3)
        sigs.append(dict(ticker=m["ticker"], kind=m["kind"], icao=m["icao"], city=s.get("site"),
                         day=m["day"], close=m["close"], strike=dict(type=m["strike_type"], floor=m["floor"],
                                                                     cap=m["cap"]),
                         p_model=round(p, 3), yes_ask=m["yes_ask"], no_ask=m["no_ask"], mid=mid,
                         edge=gate(edge(p, m["yes_ask"], m["no_ask"]), decided, m["yes_ask"], m["no_ask"]),
                         decided=decided, features=feat))
    return sigs


def run_once() -> dict:
    os.makedirs(DATA, exist_ok=True)
    t = now()
    mk = kalshi_markets()
    icaos = sorted({m["icao"] for m in mk})
    st = stations(icaos)
    ens = ensemble({i: st[i] for i in icaos if i in st})
    obs = metars(icaos)
    nb = nbm({i: st[i] for i in icaos if i in st})
    wx = {}
    for i in sorted({m["icao"] for m in mk if m["kind"] == "rain"}):
        if i in st:
            wx[i] = dict(radar_frac_25km=wms_fraction("weather_radar", "conus_base_reflectivity_mosaic",
                                                      st[i]["lat"], st[i]["lon"], 0.25),
                         radar_frac_100km=wms_fraction("weather_radar", "conus_base_reflectivity_mosaic",
                                                       st[i]["lat"], st[i]["lon"], 1.0),
                         lightning_frac_100km=wms_fraction("lightning_detection", "ldn_lightning_strike_density",
                                                           st[i]["lat"], st[i]["lon"], 1.0))
    sigs = build(mk, st, ens, obs, wx, t, nb)
    sigs.sort(key=lambda s: -s["edge"]["ev"])
    doc = dict(generated=t.isoformat(), paper_only=True, live_trading=False, n_markets=len(mk),
               n_signals=len(sigs), n_flags=sum(s["edge"]["flag"] for s in sigs),
               sources=["Kalshi public market data", "Open-Meteo ensemble (NOAA GEFS 0.25, ECMWF IFS ENS 0.25)", "NOAA NBM via Open-Meteo",
                        "AWC METAR", "NOAA nowCOAST MRMS radar + lightning"],
               note="Model probabilities are uncalibrated until scores.json has settled history. Paper only.",
               signals=sigs)
    tmp = os.path.join(DATA, "signals.json.tmp")
    json.dump(doc, open(tmp, "w"), indent=1)
    os.replace(tmp, os.path.join(DATA, "signals.json"))
    with open(os.path.join(DATA, "signals_log.jsonl"), "a") as f:
        for s in sigs:
            f.write(json.dumps(dict(t=doc["generated"], ticker=s["ticker"], kind=s["kind"], p=s["p_model"],
                                    yes_ask=s["yes_ask"], no_ask=s["no_ask"], mid=s["mid"], edge=s["edge"],
                                    hours_left=s["features"]["hours_left"])) + "\n")
    return doc


if __name__ == "__main__":
    d = run_once()
    print(f"{d['generated']} markets={d['n_markets']} signals={d['n_signals']} flags={d['n_flags']}")
