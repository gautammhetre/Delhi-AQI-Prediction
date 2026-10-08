"""Web pages. Every page calls services/core, so the website and the API behave the same way."""
from django.contrib.admin.views.decorators import staff_member_required
from django.core.paginator import Paginator
from django.db import DatabaseError
from django.shortcuts import render
from django.views.decorators.http import require_POST

from core import assistant, measures, ml, naqi
from core.report import ReportError

from . import services
from .forms import AskForm, LocationForm, PredictForm
from .models import Prediction

PROFILE_CHOICES = [
    ("general", "Me (healthy adult)"), ("children", "A child"), ("elderly", "An older adult"),
    ("lung_heart", "Asthma, lung or heart condition"), ("pregnant", "Pregnant"), ("outdoor_worker", "Outdoor worker"),
]


# ------------------------------------------------------------------ Now: live AQI for a place

def _report_context(report):
    return {
        "rep": report,
        "advice": {key: measures.for_aqi(report["aqi"]["aqi"], key) for key, _ in PROFILE_CHOICES},
        "profiles": PROFILE_CHOICES,
        "categories": naqi.CATEGORIES,
    }


def now(request):
    try:
        ctx = _report_context(services.live_report())
    except ReportError as e:
        ctx = {"error": str(e)}
    ctx["profiles"] = PROFILE_CHOICES
    return render(request, "predictor/now.html", ctx)


@require_POST
def report_partial(request):
    """HTML fragment for a chosen place. Coordinates come in the POST body, so they stay out of URLs and logs."""
    form = LocationForm(request.POST)
    if not form.is_valid():
        return render(request, "predictor/_report_error.html", {"error": "That location isn't valid."}, status=400)
    d = form.cleaned_data
    try:
        report = services.live_report(d["lat"], d["lon"], d.get("label") or None)
    except ReportError as e:
        return render(request, "predictor/_report_error.html", {"error": str(e)}, status=502)
    return render(request, "predictor/_report.html", _report_context(report))


# ------------------------------------------------------------------ Check: estimate from your own readings

def check(request):
    p = services.predictor()
    if request.method == "POST":
        form = PredictForm(request.POST, stats=p.stats)
        if form.is_valid():
            try:
                result = services.run_prediction(form.values(), profile=form.cleaned_data["profile"])
            except ValueError as e:
                form.add_error(None, str(e))
            else:
                return render(request, "predictor/result.html", {
                    "r": result, "profile": form.cleaned_data["profile"], "categories": naqi.CATEGORIES,
                })
    else:
        preset = next((x for x in p.presets if x["key"] == request.GET.get("example")), p.presets[0])
        form = PredictForm(initial={"profile": "general", **preset["values"]}, stats=p.stats)
    return render(request, "predictor/check.html", {
        "form": form, "presets": p.presets, "active_preset": request.GET.get("example", p.presets[0]["key"]),
    })


# ------------------------------------------------------------------ Guide, Ask, Trends, About

def guide(request):
    return render(request, "predictor/guide.html", {
        "rows": measures.table(), "highlight": request.GET.get("category"), "home_steps": measures.HOME_STEPS,
        "advisory": measures.ADVISORY, "grap_source": measures.GRAP_SOURCE, "vulnerable": measures.VULNERABLE_GROUPS,
    })


EXAMPLE_QUESTIONS = [
    "Is it safe for my kids to play outside at Very Poor AQI?",
    "Should I wear an N95 mask?",
    "What happens under GRAP Stage III?",
    "Do air purifiers help?",
    "Which month was worst in 2021?",
    "What time of day is pollution worst in December?",
]


def ask(request):
    answer = None
    if request.method == "POST":
        form = AskForm(request.POST)
        if form.is_valid():
            answer = assistant.ask(form.cleaned_data["question"])
    else:
        q = (request.GET.get("q") or "")[:500]
        form = AskForm(initial={"question": q} if q else None)
        if q:
            answer = assistant.ask(q)
    return render(request, "predictor/ask.html", {"form": form, "answer": answer, "examples": EXAMPLE_QUESTIONS})


def trends(request):
    return render(request, "predictor/trends.html", {"d": services.dashboard_data()})


def about(request):
    return render(request, "predictor/about.html", {"m": services.metrics()})


def model_page(request):
    m = services.metrics()
    rows, matrix, importance, cm_columns = [], [], [], []
    if m:
        rows = [{"name": name, **r, "chosen": name == m["best_model"]} for name, r in m["models"].items()]
        cm = m["confusion_matrix"]
        matrix = [{"label": lab, "cells": [{"v": v, "alpha": round(v / max(1, sum(row)), 2)} for v in row]}
                  for lab, row in zip(cm["labels"], cm["matrix"])]
        short = {"Satisfactory": "Sat.", "Moderate": "Mod.", "Very Poor": "V. Poor"}
        cm_columns = [{"short": short.get(lab, lab), "full": lab} for lab in cm["labels"]]
        top = max(m["permutation_importance_mae"].values()) or 1
        importance = [{"feature": naqi.LABELS[f], "value": v, "pct": round(100 * v / top, 1)}
                      for f, v in m["permutation_importance_mae"].items()]
    return render(request, "predictor/model.html", {"m": m, "rows": rows, "matrix": matrix, "importance": importance,
                                                    "cm_columns": cm_columns, "features": [naqi.LABELS[f] for f in ml.FEATURES]})


@staff_member_required
def history(request):
    """Every estimate made, for staff only: other people's readings aren't shown publicly."""
    try:
        page = Paginator(Prediction.objects.all(), 25).get_page(request.GET.get("page"))
        page.object_list = list(page.object_list)  # run the query here, inside the try
        db_error = None
    except DatabaseError:
        page, db_error = None, services.DB_MISSING_MESSAGE
    return render(request, "predictor/history.html", {"page": page, "db_error": db_error})
