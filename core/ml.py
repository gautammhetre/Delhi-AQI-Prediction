"""
PM2.5 model: training, evaluation and prediction.

What the model does: estimate the PM2.5 concentration for an hour in Delhi from
the other pollutants measured in that hour (CO, NO, NO2, O3, SO2, PM10, NH3).
The AQI category is then derived from the predicted PM2.5 with India's NAQI
breakpoints, so the number and the label can never disagree.

Evaluation is time-based: train on the first 80% of the timeline, test on the
most recent 20% (data the model has never seen, from later months). A random
split would let the model see hours adjacent to the test hours.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, r2_score, root_mean_squared_error

from . import naqi

FEATURES = ["co", "no", "no2", "o3", "so2", "pm10", "nh3"]
TARGET = "pm2_5"
CATEGORY_ORDER = [c[0] for c in naqi.CATEGORIES]
UNITS = "µg/m³"

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "delhi_aqi.csv"
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"


# ---------------------------------------------------------------- data

# The CSV's timestamps are UTC, not Indian time. Evidence: ozone, which forms in sunlight
# and peaks in the early afternoon, peaks at 08:00-09:00 in the raw file (13:30-14:30 IST),
# and PM2.5 is lowest at 09:00 raw (14:30 IST, when the air mixes most). The column names
# also match OpenWeather's Air Pollution API, which reports UTC. So we shift to IST.
IST_OFFSET = pd.Timedelta(hours=5, minutes=30)


def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=FEATURES + [TARGET]).drop_duplicates(subset="date")
    df["date"] = df["date"] + IST_OFFSET
    return df.sort_values("date").reset_index(drop=True)


def time_split(df: pd.DataFrame, train_frac: float = 0.8):
    cut = int(len(df) * train_frac)
    return df.iloc[:cut], df.iloc[cut:]


# ---------------------------------------------------------------- models

class PM10RatioBaseline:
    """Simplest sensible guess: PM2.5 is a fixed share of PM10 (fitted as the median ratio).
    Any real model has to beat this to be worth having."""

    def fit(self, X, y):
        X = pd.DataFrame(X, columns=FEATURES)
        ratio = np.asarray(y) / np.clip(X["pm10"].to_numpy(), 1e-6, None)
        self.ratio_ = float(np.median(ratio))
        return self

    def predict(self, X):
        X = pd.DataFrame(X, columns=FEATURES)
        return X["pm10"].to_numpy() * self.ratio_


def candidate_models() -> dict:
    return {
        "PM10 ratio (baseline)": PM10RatioBaseline(),
        "Linear regression": LinearRegression(),
        "Random forest": RandomForestRegressor(n_estimators=120, max_depth=18, min_samples_leaf=2, n_jobs=-1, random_state=42),
        "Gradient boosting": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=42),
    }


def _category_metrics(y_true, y_pred) -> dict:
    t = [naqi.pm25_category(v) for v in y_true]
    p = [naqi.pm25_category(v) for v in y_pred]
    ti = np.array([CATEGORY_ORDER.index(c) for c in t])
    pi = np.array([CATEGORY_ORDER.index(c) for c in p])
    return {
        "accuracy": round(float(accuracy_score(t, p)), 3),
        "macro_f1": round(float(f1_score(t, p, average="macro", labels=CATEGORY_ORDER, zero_division=0)), 3),
        "within_one_category": round(float(np.mean(np.abs(ti - pi) <= 1)), 3),
        "_true": t, "_pred": p,
    }


def _size_mb(model) -> float:
    import io
    buf = io.BytesIO()
    joblib.dump(model, buf, compress=3)
    return round(len(buf.getvalue()) / 1e6, 2)


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    cat = _category_metrics(y_test, pred)
    return {
        "mae": round(float(mean_absolute_error(y_test, pred)), 2),
        "rmse": round(float(root_mean_squared_error(y_test, pred)), 2),
        "r2": round(float(r2_score(y_test, pred)), 3),
        "category_accuracy": cat["accuracy"],
        "category_macro_f1": cat["macro_f1"],
        "within_one_category": cat["within_one_category"],
        "_cat": cat,
    }


def _input_stats(df: pd.DataFrame) -> dict:
    return {
        f: {
            "p1": round(float(df[f].quantile(0.01)), 2),
            "median": round(float(df[f].median()), 2),
            "p99": round(float(df[f].quantile(0.99)), 2),
            "max": round(float(df[f].max()), 2),
        }
        for f in FEATURES
    }


PRESET_RULES = [
    ("winter_night", "Winter night (Dec, 11 pm–2 am)", [12], [23, 0, 1, 2]),
    ("stubble_season", "Stubble-burning season (Nov, 6–9 pm)", [11], [18, 19, 20, 21]),
    ("spring_afternoon", "Spring afternoon (Mar, 1–4 pm)", [3], [13, 14, 15, 16]),
    ("monsoon_afternoon", "Monsoon afternoon (Aug, 1–4 pm)", [8], [13, 14, 15, 16]),
]

# Presets for specific date windows rather than whole months: (key, label, [(first day, last day)], hours)
DATE_PRESET_RULES = [
    # Diwali was on 4 Nov 2021 and 24 Oct 2022; take the 3 days either side of each.
    ("diwali_week", "Diwali week (8 pm–midnight)", [("2021-11-01", "2021-11-07"), ("2022-10-21", "2022-10-27")],
     [20, 21, 22, 23]),
]


def _preset(key: str, label: str, sub: pd.DataFrame) -> dict:
    """A preset is the median of real hours, so its estimate can be checked against what was measured."""
    values = {f: round(float(sub[f].median()), 1) for f in FEATURES}
    return {"key": key, "label": label, "values": values, "hours": int(len(sub)),
            "actual_pm25_median": round(float(sub[TARGET].median()), 1)}


def _presets(df: pd.DataFrame) -> list[dict]:
    out = []
    for key, label, months, hours in PRESET_RULES:
        sub = df[df["date"].dt.month.isin(months) & df["date"].dt.hour.isin(hours)]
        if not sub.empty:
            out.append(_preset(key, label, sub))
    for key, label, windows, hours in DATE_PRESET_RULES:
        day = df["date"].dt.normalize()
        in_window = pd.Series(False, index=df.index)
        for first, last in windows:
            in_window |= day.between(pd.Timestamp(first), pd.Timestamp(last))
        sub = df[in_window & df["date"].dt.hour.isin(hours)]
        if not sub.empty:
            out.append(_preset(key, label, sub))
    return out


def train_and_evaluate(df: pd.DataFrame | None = None, log=print) -> tuple[dict, dict]:
    """Returns (artifact, metrics). The chosen model is refit on all data after evaluation."""
    df = load_data() if df is None else df
    train, test = time_split(df)
    Xtr, ytr, Xte, yte = train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]

    results, fitted = {}, {}
    for name, model in candidate_models().items():
        t0 = time.time()
        model.fit(Xtr, ytr)
        r = evaluate(model, Xte, yte)
        r["train_seconds"] = round(time.time() - t0, 1)
        r["size_mb"] = _size_mb(model)
        results[name], fitted[name] = r, model
        log(f"  {name:<24} MAE {r['mae']:>7.2f}  R² {r['r2']:.3f}  category acc {r['category_accuracy']:.3f}  {r['size_mb']} MB")

    # Selection rule: among real models, take the lowest MAE, but if a smaller model is within
    # 5% of that MAE prefer it. The app has to load the model on a free hosting tier.
    real = [n for n in results if "baseline" not in n]
    lowest = min(results[n]["mae"] for n in real)
    close = [n for n in real if results[n]["mae"] <= lowest * 1.05]
    best_name = min(close, key=lambda n: results[n]["size_mb"])
    best = fitted[best_name]
    log(f"  -> chosen: {best_name} (lowest MAE {lowest}, chosen MAE {results[best_name]['mae']}, {results[best_name]['size_mb']} MB)")

    # Ablation: how much do the other gases add beyond PM10? Train the same model type without PM10.
    no_pm10 = [f for f in FEATURES if f != "pm10"]
    abl = candidate_models()[best_name].fit(train[no_pm10], ytr)
    abl_pred = abl.predict(test[no_pm10])
    ablation = {"features": no_pm10, "mae": round(float(mean_absolute_error(yte, abl_pred)), 2),
                "r2": round(float(r2_score(yte, abl_pred)), 3)}

    # Which inputs matter, measured on the test set: shuffle one feature and see how much MAE gets worse.
    sample = test.sample(n=min(3000, len(test)), random_state=0)
    perm = permutation_importance(best, sample[FEATURES], sample[TARGET], scoring="neg_mean_absolute_error",
                                  n_repeats=5, random_state=0, n_jobs=-1)
    importance = {f: round(float(v), 2) for f, v in sorted(zip(FEATURES, perm.importances_mean), key=lambda kv: -kv[1])}

    cat = results[best_name]["_cat"]
    cm = confusion_matrix(cat["_true"], cat["_pred"], labels=CATEGORY_ORDER).tolist()

    metrics = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "rows": int(len(df)),
        "date_range": [str(df["date"].min()), str(df["date"].max())],
        "train": {"rows": int(len(train)), "from": str(train["date"].min()), "to": str(train["date"].max())},
        "test": {"rows": int(len(test)), "from": str(test["date"].min()), "to": str(test["date"].max())},
        "models": {n: {k: v for k, v in r.items() if not k.startswith("_")} for n, r in results.items()},
        "best_model": best_name,
        "selection_rule": "Lowest test MAE among real models; a smaller model within 5% of that MAE is preferred.",
        "ablation_without_pm10": ablation,
        "permutation_importance_mae": importance,
        "confusion_matrix": {"labels": CATEGORY_ORDER, "matrix": cm},
        "test_category_share": {c: round(cat["_true"].count(c) / len(cat["_true"]), 3) for c in CATEGORY_ORDER},
        "note": "Categories use NAQI PM2.5 breakpoints on hourly values (the official index uses 24-hour averages). "
                "Timestamps converted from UTC to IST.",
    }

    final = candidate_models()[best_name].fit(df[FEATURES], df[TARGET])
    artifact = {
        "model": final,
        "model_name": best_name,
        "features": FEATURES,
        "sklearn_version": sklearn.__version__,
        "trained_at": metrics["trained_at"],
        "input_stats": _input_stats(df),
        "presets": _presets(df),
        "test_mae": results[best_name]["mae"],
    }
    return artifact, metrics


def save(artifact: dict, metrics: dict | None = None, model_path: Path = MODEL_PATH,
         metrics_path: Path = METRICS_PATH):
    """The model is pickled on its own (model.joblib); everything else is plain JSON
    (model_meta.json), so the presets, input ranges and metrics stay readable on any
    scikit-learn version even when the pickle itself can't be loaded."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in artifact.items() if k != "model"}
    meta["sklearn_version"] = sklearn.__version__
    meta["check"] = [{"values": p["values"], "pred": float(artifact["model"].predict(
        pd.DataFrame([p["values"]])[FEATURES])[0])} for p in artifact["presets"]]
    joblib.dump({"model": artifact["model"], "sklearn_version": sklearn.__version__}, model_path, compress=3)
    _meta_path(model_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if metrics is not None:
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _meta_path(model_path: Path) -> Path:
    return model_path.with_name("model_meta.json")


def refit_final_model(meta: dict, log=print):
    """Refit only the chosen model on all the data (a few seconds). Used when the saved pickle
    was made with a different scikit-learn version: pickles aren't portable across versions."""
    t0 = time.time()
    df = load_data()
    model = candidate_models()[meta["model_name"]].fit(df[FEATURES], df[TARGET])
    log(f"[ml] Refitted {meta['model_name']} for scikit-learn {sklearn.__version__} in {time.time() - t0:.1f}s")
    return model


# ---------------------------------------------------------------- prediction

@dataclass
class Prediction:
    pm25: float
    pm25_category: str
    aqi: dict
    contributions: list
    warnings: list
    model_name: str
    test_mae: float

    def as_dict(self) -> dict:
        return {
            "pm25": self.pm25,
            "pm25_category": self.pm25_category,
            "aqi": self.aqi,
            "contributions": self.contributions,
            "warnings": self.warnings,
            "model": self.model_name,
            "typical_error_ugm3": self.test_mae,
        }


class Predictor:
    def __init__(self, artifact: dict):
        self.artifact = artifact
        self.model = artifact["model"]
        self.stats = artifact["input_stats"]

    @classmethod
    def load(cls, path: Path = MODEL_PATH, log=print) -> "Predictor":
        """Load the saved model. If the pickle is missing, unreadable, or was made with another
        scikit-learn version, refit the chosen model (seconds, not the full evaluation) and save it.
        If even the metadata is missing, run the full training once."""
        meta_path = _meta_path(path)
        if not meta_path.exists():
            log("[ml] No saved model found. Training and evaluating, this takes under a minute...")
            artifact, metrics = train_and_evaluate(log=log)
            save(artifact, metrics, path, path.parent / "metrics.json")
            return cls(artifact)

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model, reason = None, ""
        try:
            saved = joblib.load(path)
            if saved.get("sklearn_version") != sklearn.__version__:
                reason = f"saved with scikit-learn {saved.get('sklearn_version')}, running {sklearn.__version__}"
            else:
                model = saved["model"]
        except Exception as e:  # noqa: BLE001 - any load problem means refit
            reason = f"{e.__class__.__name__}: {e}"
        if model is not None and not cls._passes_check(model, meta):
            model, reason = None, "saved model gives different predictions than when it was trained"
        if model is None:
            log(f"[ml] Can't use the saved model file ({reason}). Refitting it...")
            model = refit_final_model(meta, log)
            artifact = {**meta, "model": model}
            save(artifact, None, path)
            return cls(artifact)
        return cls({**meta, "model": model})

    @staticmethod
    def _passes_check(model, meta) -> bool:
        for c in meta.get("check", []):
            got = float(model.predict(pd.DataFrame([c["values"]])[FEATURES])[0])
            if abs(got - c["pred"]) > max(0.5, 0.01 * abs(c["pred"])):
                return False
        return True

    @property
    def presets(self):
        return self.artifact["presets"]

    def _frame(self, inputs: dict) -> pd.DataFrame:
        return pd.DataFrame([[float(inputs[f]) for f in FEATURES]], columns=FEATURES)

    def range_warnings(self, inputs: dict) -> list[str]:
        out = []
        for f in FEATURES:
            v, s = float(inputs[f]), self.stats[f]
            if v > s["max"]:
                out.append(f"{naqi.LABELS[f]} = {v:g} is above anything in the training data (max {s['max']:g}); treat the prediction with caution.")
            elif v < s["p1"] or v > s["p99"]:
                out.append(f"{naqi.LABELS[f]} = {v:g} is unusual for Delhi (98% of hours are between {s['p1']:g} and {s['p99']:g}).")
        return out

    def explain(self, inputs: dict, base_pred: float) -> list[dict]:
        """How much each input moves the prediction compared with a typical Delhi hour:
        replace one input with its median and measure the change."""
        rows = []
        for f in FEATURES:
            alt = dict(inputs)
            alt[f] = self.stats[f]["median"]
            delta = base_pred - float(self.model.predict(self._frame(alt))[0])
            rows.append({"feature": f, "label": naqi.LABELS[f], "value": float(inputs[f]),
                         "typical": self.stats[f]["median"], "effect": round(delta, 1)})
        return sorted(rows, key=lambda r: -abs(r["effect"]))

    def predict(self, inputs: dict) -> Prediction:
        missing = [f for f in FEATURES if inputs.get(f) is None]
        if missing:
            raise ValueError(f"missing inputs: {', '.join(missing)}")
        if any(float(inputs[f]) < 0 for f in FEATURES):
            raise ValueError("concentrations cannot be negative")
        pm25 = max(0.0, float(self.model.predict(self._frame(inputs))[0]))
        conc = {f: float(inputs[f]) for f in FEATURES if f in naqi.BREAKPOINTS}
        conc["pm2_5"] = pm25
        aqi = naqi.overall_aqi(conc)
        return Prediction(
            pm25=round(pm25, 1),
            pm25_category=naqi.pm25_category(pm25),
            aqi=aqi,
            contributions=self.explain(inputs, pm25),
            warnings=self.range_warnings(inputs),
            model_name=self.artifact["model_name"],
            test_mae=self.artifact["test_mae"],
        )


def load_metrics(path: Path = METRICS_PATH) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
