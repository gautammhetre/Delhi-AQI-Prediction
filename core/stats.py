"""
Aggregates of the historical Delhi data, used by the dashboard and as tools the
assistant can call ("Which month was worst in 2021?"). Every function returns
plain JSON-serialisable data and states the period it covers, so neither the
page nor the LLM can claim more than the data supports.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from . import ml, naqi

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Diwali dates inside the dataset's range, for the dashboard annotation.
DIWALI = {"2021": "2021-11-04", "2022": "2022-10-24"}


@lru_cache(maxsize=1)
def _df() -> pd.DataFrame:
    df = ml.load_data()
    df["day"] = df["date"].dt.normalize()
    return df


def coverage() -> dict:
    df = _df()
    return {"from": str(df["date"].min().date()), "to": str(df["date"].max().date()), "hours": int(len(df)),
            "years": sorted({int(y) for y in df["date"].dt.year.unique()})}


def _r(x):
    """Round for display; missing values become None so the JSON stays valid (no NaN)."""
    if x is None or pd.isna(x):
        return None
    return round(float(x), 1)


def dashboard() -> dict:
    df = _df()
    by_month = df.groupby(df["date"].dt.to_period("M"))[ml.TARGET].mean()
    moy = df.groupby(df["date"].dt.month)[ml.TARGET].mean()
    hod = df.groupby(df["date"].dt.hour)[ml.TARGET].mean()
    winter = df[df["date"].dt.month.isin([11, 12, 1])]
    summer = df[df["date"].dt.month.isin([6, 7, 8])]
    hod_winter = winter.groupby(winter["date"].dt.hour)[ml.TARGET].mean()
    hod_summer = summer.groupby(summer["date"].dt.hour)[ml.TARGET].mean()
    cats = df[ml.TARGET].map(naqi.pm25_category).value_counts(normalize=True)
    daily = df.groupby("day")[ml.TARGET].mean()

    diwali = []
    for year, d in DIWALI.items():
        day = pd.Timestamp(d)
        if day in daily.index:
            month = daily[(daily.index.month == day.month) & (daily.index.year == day.year)]
            diwali.append({"year": year, "date": d, "day_after": _r(daily.get(day + pd.Timedelta(days=1), float("nan"))),
                           "diwali_day": _r(daily[day]), "month_average": _r(month.mean())})
    return {
        "coverage": coverage(),
        "monthly_series": {"labels": [str(p) for p in by_month.index], "values": [_r(v) for v in by_month.values]},
        "month_of_year": {"labels": [MONTHS[m - 1] for m in moy.index], "values": [_r(v) for v in moy.values]},
        "hour_of_day": {"labels": [f"{h:02d}:30" for h in hod.index], "all": [_r(v) for v in hod.values],
                        "winter": [_r(hod_winter.get(h)) for h in hod.index], "summer": [_r(hod_summer.get(h)) for h in hod.index]},
        "category_share": {c: round(float(cats.get(c, 0)), 3) for c in ml.CATEGORY_ORDER},
        "worst_days": worst_days(5)["days"],
        "diwali": diwali,
        "overall_mean": _r(df[ml.TARGET].mean()),
        "who_24h_guideline": 15,
    }


# ------------------------------------------------------------------ assistant tools

def monthly_average(year: int | None = None, month: int | None = None) -> dict:
    """Average PM2.5 for a calendar month, optionally restricted to one year."""
    df = _df()
    sel = df
    if year is not None:
        sel = sel[sel["date"].dt.year == int(year)]
    if month is not None:
        sel = sel[sel["date"].dt.month == int(month)]
    if sel.empty:
        return {"error": "No data for that period.", "coverage": coverage()}
    mean = float(sel[ml.TARGET].mean())
    return {"year": year, "month": MONTHS[int(month) - 1] if month else None, "hours": int(len(sel)),
            "mean_pm25": _r(mean), "pm25_category_of_mean": naqi.pm25_category(mean),
            "share_severe_hours": round(float((sel[ml.TARGET] > 250).mean()), 3), "unit": ml.UNITS}


def worst_days(n: int = 5, year: int | None = None) -> dict:
    """Days with the highest daily-average PM2.5."""
    df = _df()
    if year is not None:
        df = df[df["date"].dt.year == int(year)]
    if df.empty:
        return {"error": "No data for that year.", "coverage": coverage()}
    daily = df.groupby("day")[ml.TARGET].mean().sort_values(ascending=False).head(max(1, min(int(n), 20)))
    return {"year": year, "unit": ml.UNITS,
            "days": [{"date": str(d.date()), "mean_pm25": _r(v), "category": naqi.pm25_category(v)} for d, v in daily.items()]}


def hourly_profile(month: int | None = None) -> dict:
    """Average PM2.5 by hour of day, optionally for one calendar month."""
    df = _df()
    if month is not None:
        df = df[df["date"].dt.month == int(month)]
    if df.empty:
        return {"error": "No data for that month."}
    hod = df.groupby(df["date"].dt.hour)[ml.TARGET].mean()
    # Readings are hourly at hh:30 IST (UTC on the hour, shifted by 5:30).
    return {"month": MONTHS[int(month) - 1] if month else "all", "unit": ml.UNITS,
            "worst_hour": f"{int(hod.idxmax()):02d}:30", "best_hour": f"{int(hod.idxmin()):02d}:30",
            "by_hour": {f"{h:02d}:30": _r(v) for h, v in hod.items()}}


def year_summary(year: int) -> dict:
    """Mean PM2.5 for a year, its worst and best months, and how much of the year was Severe."""
    df = _df()
    sel = df[df["date"].dt.year == int(year)]
    if sel.empty:
        return {"error": f"No data for {year}.", "coverage": coverage()}
    by_m = sel.groupby(sel["date"].dt.month)[ml.TARGET].mean()
    return {"year": int(year), "hours": int(len(sel)), "months_covered": [MONTHS[m - 1] for m in by_m.index],
            "mean_pm25": _r(sel[ml.TARGET].mean()),
            "worst_month": {"month": MONTHS[int(by_m.idxmax()) - 1], "mean_pm25": _r(by_m.max())},
            "best_month": {"month": MONTHS[int(by_m.idxmin()) - 1], "mean_pm25": _r(by_m.min())},
            "share_severe_hours": round(float((sel[ml.TARGET] > 250).mean()), 3), "unit": ml.UNITS,
            "note": "Partial year" if len(by_m) < 12 else "Full year"}
