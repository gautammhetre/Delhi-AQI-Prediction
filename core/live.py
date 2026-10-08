"""
Current pollutant readings from the Open-Meteo Air Quality API (free, no key, non-commercial
use). Values are model estimates (CAMS), in µg/m³. Works for central Delhi or any coordinates,
e.g. the browser's current location.

* Cached for 10 minutes per location (rounded to ~1 km) so refreshes don't hit the API each time.
* Ammonia is only published for Europe, so elsewhere the typical Delhi value is used and the
  response lists it under "estimated".
* If the API can't be reached, the response falls back to a typical Delhi hour and says so,
  so the page keeps working.
* The response says how far the location is from Delhi: the model was trained on Delhi only.
"""
from __future__ import annotations

import math
import time

import requests

URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
DELHI = (28.6139, 77.2090)
FIELDS = {  # open-meteo name -> our feature name
    "carbon_monoxide": "co", "nitrogen_monoxide": "no", "nitrogen_dioxide": "no2", "ozone": "o3",
    "sulphur_dioxide": "so2", "pm10": "pm10", "ammonia": "nh3", "pm2_5": "pm2_5",
}
TTL_SECONDS = 600
NEAR_DELHI_KM = 60  # roughly the NCR
_cache: dict = {}


def distance_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle (haversine) distance."""
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))


def validate_coords(lat, lon) -> tuple[float, float]:
    lat, lon = float(lat), float(lon)
    if not (-90 <= lat <= 90 and -180 <= lon <= 180) or math.isnan(lat) or math.isnan(lon):
        raise ValueError("latitude must be -90..90 and longitude -180..180")
    return lat, lon


def fetch_current(medians: dict, fallback: dict, lat: float | None = None, lon: float | None = None,
                  get=None, now=None) -> dict:
    """medians: typical value per feature (used for missing ones). fallback: values if the API fails.
    lat/lon default to central Delhi."""
    is_delhi = lat is None or lon is None
    lat, lon = DELHI if is_delhi else validate_coords(lat, lon)
    key = (round(lat, 2), round(lon, 2))
    dist = round(distance_km(DELHI, (lat, lon)), 1)
    place = {"lat": key[0], "lon": key[1], "label": "Central Delhi" if is_delhi else "Your location",
             "distance_from_delhi_km": dist, "near_delhi": dist <= NEAR_DELHI_KM}

    now = now or time.time()
    hit = _cache.get(key)
    if hit and now - hit["at"] < TTL_SECONDS:
        return {**hit["data"], **{"place": place}, "cached": True}

    get = get or requests.get
    params = {"latitude": key[0], "longitude": key[1], "current": ",".join(FIELDS), "timezone": "auto"}
    try:
        res = get(URL, params=params, timeout=8)
        res.raise_for_status()
        cur = res.json()["current"]
        values, estimated = {}, []
        for api_name, f in FIELDS.items():
            if f == "pm2_5":
                continue
            v = cur.get(api_name)
            if v is None:
                values[f] = medians[f]
                estimated.append(f)
            else:
                values[f] = round(float(v), 1)
        data = {"source": "open-meteo", "time": cur.get("time"),
                "time_display": str(cur.get("time") or "").replace("T", " "), "values": values, "estimated": estimated,
                "reported_pm25": cur.get("pm2_5"), "cached": False, "place": place}
        _cache[key] = {"at": now, "data": data}
        return data
    except (requests.RequestException, KeyError, ValueError, TypeError) as e:
        return {"source": "fallback", "time": None, "time_display": "", "values": dict(fallback), "estimated": list(fallback),
                "reported_pm25": None, "cached": False, "place": place,
                "error": f"Live data unavailable ({e.__class__.__name__}); showing a typical Delhi hour instead."}


def clear_cache():
    _cache.clear()
