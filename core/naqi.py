"""
India's National Air Quality Index (NAQI), as published by the Central Pollution
Control Board (CPCB).

Each pollutant concentration is converted to a sub-index on a 0-500 scale by
linear interpolation inside its breakpoint band. The overall AQI is the highest
sub-index, and the pollutant that produced it is the "prominent pollutant".

Breakpoints are defined for 24-hour averages (8-hour for CO and O3). This app
works with hourly readings, so the AQI it shows is an indicative hourly value,
not the official daily AQI. The UI says so.

This module has no Django or ML imports so it can be unit-tested on its own.
"""
from __future__ import annotations

CATEGORIES = [
    # name, AQI low, AQI high, health impact (CPCB wording, paraphrased)
    ("Good", 0, 50, "Minimal impact."),
    ("Satisfactory", 51, 100, "Minor breathing discomfort for sensitive people."),
    ("Moderate", 101, 200, "Breathing discomfort for people with lung or heart disease, children and older adults."),
    ("Poor", 201, 300, "Breathing discomfort for most people on prolonged exposure."),
    ("Very Poor", 301, 400, "Respiratory illness on prolonged exposure."),
    ("Severe", 401, 500, "Affects healthy people and seriously affects those with existing diseases."),
]

# Concentration breakpoints per category, in the same order as CATEGORIES.
# Units: µg/m³, except CO in mg/m³. The last band is open-ended.
BREAKPOINTS = {
    "pm2_5": [(0, 30), (31, 60), (61, 90), (91, 120), (121, 250), (251, 380)],
    "pm10":  [(0, 50), (51, 100), (101, 250), (251, 350), (351, 430), (431, 510)],
    "no2":   [(0, 40), (41, 80), (81, 180), (181, 280), (281, 400), (401, 520)],
    "o3":    [(0, 50), (51, 100), (101, 168), (169, 208), (209, 748), (749, 1288)],
    "co":    [(0, 1.0), (1.1, 2.0), (2.1, 10), (10.1, 17), (17.1, 34), (34.1, 51)],
    "so2":   [(0, 40), (41, 80), (81, 380), (381, 800), (801, 1600), (1601, 2400)],
    "nh3":   [(0, 200), (201, 400), (401, 800), (801, 1200), (1201, 1800), (1801, 2400)],
}
# The upper value of each "Severe" band above is an extrapolation used only to
# keep the interpolation finite; anything beyond it is capped at AQI 500.

LABELS = {
    "pm2_5": "PM2.5", "pm10": "PM10", "no2": "NO₂", "o3": "O₃",
    "co": "CO", "so2": "SO₂", "nh3": "NH₃", "no": "NO",
}

# GRAP (Graded Response Action Plan) for Delhi-NCR, CAQM. Stages are triggered by AQI.
GRAP_STAGES = [
    ("Stage I", 201, 300, "Poor"),
    ("Stage II", 301, 400, "Very Poor"),
    ("Stage III", 401, 450, "Severe"),
    ("Stage IV", 451, 10_000, "Severe+"),
]


def category_for_aqi(aqi: float) -> dict:
    a = max(0, round(aqi))
    for i, (name, lo, hi, health) in enumerate(CATEGORIES):
        if a <= hi or i == len(CATEGORIES) - 1:
            return {"name": name, "low": lo, "high": hi, "health": health, "index": i}


def pm25_category(conc: float) -> str:
    """Category from a PM2.5 concentration alone (µg/m³). Used for model evaluation."""
    return category_for_aqi(sub_index("pm2_5", conc))["name"]


def sub_index(pollutant: str, conc: float) -> float:
    """Linear interpolation inside the breakpoint band that contains `conc`.

    CPCB bands have small gaps (30 then 31). A value inside a gap, e.g. 30.5,
    belongs to the band whose upper limit it exceeds, so it is placed at the
    start of the next band.
    """
    if conc is None:
        raise ValueError("concentration is required")
    c = max(0.0, float(conc))
    bands = BREAKPOINTS[pollutant]
    for i, (lo, hi) in enumerate(bands):
        if c <= hi or i == len(bands) - 1:
            i_lo, i_hi = CATEGORIES[i][1], CATEGORIES[i][2]
            c_eff = min(max(c, lo), hi)  # gap values snap to the band start; beyond the last band caps at 500
            value = i_lo + (i_hi - i_lo) * (c_eff - lo) / (hi - lo)
            return round(value, 1)
    return 500.0  # unreachable: the last band always matches


def overall_aqi(concentrations: dict) -> dict:
    """concentrations: {pollutant: value}, CO in µg/m³ (it is converted to mg/m³ here).

    Returns AQI, category, prominent pollutant and every sub-index.
    CPCB needs at least three pollutants, one of them PM2.5 or PM10.
    """
    subs = {}
    for p, v in concentrations.items():
        if p not in BREAKPOINTS or v is None:
            continue
        value = v / 1000.0 if p == "co" else v
        subs[p] = sub_index(p, value)
    if len(subs) < 3 or not ({"pm2_5", "pm10"} & subs.keys()):
        raise ValueError("NAQI needs at least three pollutants including PM2.5 or PM10")
    prominent = max(subs, key=subs.get)
    aqi = subs[prominent]
    return {
        "aqi": round(aqi),
        "category": category_for_aqi(aqi),
        "prominent": prominent,
        "prominent_label": LABELS[prominent],
        "sub_indices": {p: round(s) for p, s in sorted(subs.items(), key=lambda kv: -kv[1])},
        "grap": grap_stage(aqi),
    }


def grap_stage(aqi: float) -> dict | None:
    a = round(aqi)
    for stage, lo, hi, label in GRAP_STAGES:
        if lo <= a <= hi:
            return {"stage": stage, "low": lo, "high": hi, "label": label}
    return None
