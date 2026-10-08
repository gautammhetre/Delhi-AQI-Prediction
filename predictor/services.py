"""
The glue between Django and the core logic. Views and API views both call these
functions, so the website and the API always behave the same way.
"""
import logging
from functools import lru_cache

from django.db import DatabaseError

from core import assistant, live, measures, ml, report, stats

from .models import Prediction

log = logging.getLogger("predictor")


@lru_cache(maxsize=1)
def predictor() -> ml.Predictor:
    """Loaded once per process on first use, not on every request.
    (Loading at import time would also slow down every manage.py command.)"""
    return ml.Predictor.load(log=log.info)


def reset_predictor():
    predictor.cache_clear()


def run_prediction(values: dict, source: str = Prediction.Source.WEB, profile: str = "general",
                   explain: bool = True) -> dict:
    pred = predictor().predict(values).as_dict()
    try:
        record = Prediction.objects.create(
            source=source, **{f: float(values[f]) for f in ml.FEATURES},
            predicted_pm25=pred["pm25"], aqi=pred["aqi"]["aqi"], category=pred["aqi"]["category"]["name"],
            prominent=pred["aqi"]["prominent"], model_name=pred["model"],
        )
        pred["id"] = record.id
    except DatabaseError as e:
        # Most often: the tables were never created. The prediction itself still works.
        log.warning("Could not save prediction: %s", e)
        pred["id"] = None
        pred["warnings"] = pred["warnings"] + [DB_MISSING_MESSAGE]
    pred["measures"] = measures.for_aqi(pred["aqi"]["aqi"], profile)
    if explain:
        pred["explanation"] = assistant.explain(pred, profile)
    return pred


DB_MISSING_MESSAGE = ("This estimate was not saved to History because the database isn't set up. "
                      "Stop the server and run: python manage.py migrate")


def live_readings(lat: float | None = None, lon: float | None = None) -> dict:
    """Current readings for central Delhi, or for the given coordinates (e.g. the browser's location)."""
    p = predictor()
    medians = {f: p.stats[f]["median"] for f in ml.FEATURES}
    fallback = next((x["values"] for x in p.presets if x["key"] == "winter_night"), medians)
    return live.fetch_current(medians, fallback, lat=lat, lon=lon)


def live_report(lat: float | None = None, lon: float | None = None, label: str | None = None) -> dict:
    """Live AQI for a place (central Delhi by default). Raises report.ReportError if data is unavailable."""
    return report.fetch_report(lat, lon, label)


def search_places(q: str) -> list[dict]:
    return report.search_places(q)


@lru_cache(maxsize=1)
def dashboard_data() -> dict:
    return stats.dashboard()


def metrics() -> dict | None:
    return ml.load_metrics()
