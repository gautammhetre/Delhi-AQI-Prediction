"""
Live air-quality report for any place, built the way CPCB defines the AQI.

Data: Open-Meteo Air Quality API (CAMS model, free, no key), hourly values for the past day
and the next two days.

How the AQI is calculated (closer to the official method than a single hourly reading):
  * PM2.5, PM10, NO2, SO2: average of the last 24 hours
  * O3 and CO: average of the last 8 hours
  * NH3 is only published for Europe, so elsewhere it is left out (never filled in with a guess)
  * NAQI needs at least 3 pollutants including PM2.5 or PM10; the worst sub-index is the AQI

Outlook: for each of the next 24 hours, an indicative level from that hour's PM2.5 and PM10.
"Best time to go out" is the cleanest 2-hour window between 6 am and 10 pm.

If the data can't be fetched, the report says so. It never shows made-up numbers as live data.
"""
from __future__ import annotations

import time
from datetime import datetime

import requests

from . import naqi
from .live import DELHI, NEAR_DELHI_KM, distance_km, validate_coords

URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
HOURLY = {  # open-meteo name -> our key
    "pm2_5": "pm2_5", "pm10": "pm10", "nitrogen_dioxide": "no2", "sulphur_dioxide": "so2",
    "ozone": "o3", "carbon_monoxide": "co", "ammonia": "nh3",
}
WINDOW_HOURS = {"pm2_5": 24, "pm10": 24, "no2": 24, "so2": 24, "nh3": 24, "o3": 8, "co": 8}
SHOW = ["pm2_5", "pm10", "no2", "o3", "so2", "co"]
TTL_SECONDS = 600
_cache: dict = {}
_geo_cache: dict = {}


class ReportError(RuntimeError):
    pass


def _avg(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    # at least two-thirds of the window must be present, like CPCB's 16-of-24-hours rule
    return sum(vals) / len(vals) if vals and len(vals) >= 0.66 * len(values) else None


def _hour_label(hour: int) -> str:
    """6 -> '6 am', 0 -> '12 am'. Built by hand: strftime's %-I doesn't work on Windows."""
    return f"{(hour % 12) or 12} {'am' if hour < 12 else 'pm'}"


def build_report(payload: dict, lat: float, lon: float, label: str) -> dict:
    h = payload["hourly"]
    times = [datetime.fromisoformat(t) for t in h["time"]]
    now_str = payload.get("current", {}).get("time")
    now = datetime.fromisoformat(now_str) if now_str else max(t for t in times if t <= datetime.now())
    idx_now = max(i for i, t in enumerate(times) if t <= now)

    conc, pollutants = {}, []
    for api_name, key in HOURLY.items():
        series = h.get(api_name) or []
        w = WINDOW_HOURS[key]
        window = series[max(0, idx_now - w + 1): idx_now + 1]
        value = _avg(window) if len(window) == w else None
        if value is None:
            continue
        conc[key] = value
    try:
        aqi = naqi.overall_aqi(conc)
    except ValueError as e:
        raise ReportError(f"not enough pollutant data for this place ({e})") from e

    for key in SHOW:
        if key not in conc:
            continue
        shown = conc[key] / 1000 if key == "co" else conc[key]
        pollutants.append({
            "key": key, "label": naqi.LABELS[key], "value": round(shown, 2) if key == "co" else int(round(shown)),
            "unit": "mg/m³" if key == "co" else "µg/m³", "window": f"{WINDOW_HOURS[key]}-hour average",
            "sub_index": aqi["sub_indices"].get(key), "category": naqi.category_for_aqi(aqi["sub_indices"].get(key, 0))["name"],
        })

    outlook = []
    for i in range(idx_now + 1, min(idx_now + 25, len(times))):
        pm25, pm10 = h["pm2_5"][i], h["pm10"][i]
        if pm25 is None or pm10 is None:
            continue
        level = max(naqi.sub_index("pm2_5", pm25), naqi.sub_index("pm10", pm10))
        outlook.append({"time": times[i].strftime("%Y-%m-%dT%H:%M"), "hour": _hour_label(times[i].hour),
                        "aqi": round(level), "category": naqi.category_for_aqi(level)["name"]})

    best = None
    daytime = [o for o in outlook if 6 <= datetime.fromisoformat(o["time"]).hour <= 21]
    for a, b in zip(daytime, daytime[1:]):
        if (datetime.fromisoformat(b["time"]) - datetime.fromisoformat(a["time"])).seconds != 3600:
            continue
        score = (a["aqi"] + b["aqi"]) / 2
        if best is None or score < best["aqi"]:
            end = datetime.fromisoformat(b["time"]).hour + 1
            best = {"from": a["hour"], "to": _hour_label(end % 24),
                    "aqi": round(score), "category": naqi.category_for_aqi(score)["name"]}
    worst = max(outlook, key=lambda o: o["aqi"]) if outlook else None

    dist = round(distance_km(DELHI, (lat, lon)), 1)
    return {
        "place": {"label": label, "lat": round(lat, 2), "lon": round(lon, 2),
                  "distance_from_delhi_km": dist, "in_delhi_ncr": dist <= NEAR_DELHI_KM},
        "updated": now.strftime("%Y-%m-%dT%H:%M"),
        "updated_display": f"{(now.hour % 12) or 12}:{now.minute:02d} {'am' if now.hour < 12 else 'pm'}, {now.day} {now.strftime('%b')}",
        "outlook_peak": max((o["aqi"] for o in outlook), default=0),
        "timezone": payload.get("timezone"),
        "aqi": aqi,
        "pollutants": pollutants,
        "missing": [naqi.LABELS[k] for k in HOURLY.values() if k not in conc],
        "outlook": outlook,
        "best_window": best,
        "worst_hour": worst,
        "source": {"name": "Open-Meteo Air Quality (CAMS)", "url": "https://open-meteo.com/en/docs/air-quality-api",
                   "note": "Modelled values, not a ground monitoring station."},
    }


def fetch_report(lat: float | None = None, lon: float | None = None, label: str | None = None,
                 get=None, now=None) -> dict:
    lat, lon = (DELHI if lat is None or lon is None else validate_coords(lat, lon))
    label = (label or ("New Delhi" if (lat, lon) == DELHI else "Your location"))[:80]
    key = (round(lat, 2), round(lon, 2))
    now = now or time.time()
    hit = _cache.get(key)
    if hit and now - hit["at"] < TTL_SECONDS:
        rep = dict(hit["report"])
        rep["place"] = {**rep["place"], "label": label}
        return rep
    get = get or requests.get
    params = {"latitude": key[0], "longitude": key[1], "hourly": ",".join(HOURLY), "current": "pm2_5",
              "timezone": "auto", "past_days": 1, "forecast_days": 2}
    try:
        res = get(URL, params=params, timeout=10)
        res.raise_for_status()
        report = build_report(res.json(), lat, lon, label)
    except ReportError:
        raise
    except (requests.RequestException, KeyError, ValueError, TypeError) as e:
        raise ReportError(f"live data is unavailable right now ({e.__class__.__name__})") from e
    _cache[key] = {"at": now, "report": report}
    return report


def search_places(query: str, get=None) -> list[dict]:
    """City search via Open-Meteo's free geocoding API, India first."""
    q = " ".join((query or "").split())[:60]
    if len(q) < 2:
        return []
    if q.lower() in _geo_cache:
        return _geo_cache[q.lower()]
    get = get or requests.get
    try:
        res = get(GEOCODE_URL, params={"name": q, "count": 8, "language": "en", "format": "json"}, timeout=8)
        res.raise_for_status()
        rows = res.json().get("results") or []
    except (requests.RequestException, ValueError):
        return []
    out = []
    for r in rows:
        parts = [r.get("name"), r.get("admin1"), r.get("country")]
        out.append({"label": ", ".join(p for p in parts if p), "name": r.get("name"),
                    "lat": round(float(r["latitude"]), 4), "lon": round(float(r["longitude"]), 4),
                    "country_code": r.get("country_code")})
    out.sort(key=lambda r: r["country_code"] != "IN")  # Indian places first
    _geo_cache[q.lower()] = out
    return out


def clear_cache():
    _cache.clear()
    _geo_cache.clear()
