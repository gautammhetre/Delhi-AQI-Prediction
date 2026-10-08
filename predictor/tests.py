"""Django tests: pages, form validation, database writes and the REST API.
Run: python manage.py test   (the core logic has its own tests: python -m unittest discover tests_core)

The LLM is switched off for these tests (LLM_API_KEY=""), so nothing calls a paid API.
"""
import os
from unittest import mock

from django.core.cache import cache
from django.db import OperationalError
from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient

from core import ml

from . import services
from .models import Prediction


def sample_values():
    return dict(services.predictor().presets[0]["values"])


def fake_payload(pm25=80.0, pm10=200.0, o3=40.0, hours=72, now_index=30):
    """Open-Meteo-shaped hourly payload, so page tests never touch the network."""
    from datetime import datetime, timedelta
    t0 = datetime(2026, 10, 7, 0, 0)
    times = [(t0 + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(hours)]
    flat = lambda v: [v] * hours
    return {"timezone": "Asia/Kolkata", "current": {"time": times[now_index]},
            "hourly": {"time": times, "pm2_5": [pm25 + (i % 24) for i in range(hours)], "pm10": flat(pm10),
                       "nitrogen_dioxide": flat(30.0), "sulphur_dioxide": flat(10.0), "ozone": flat(o3),
                       "carbon_monoxide": flat(800.0), "ammonia": [None] * hours}}


def fake_report(lat=None, lon=None, label=None):
    from core import report
    return report.build_report(fake_payload(), lat or 28.61, lon or 77.21, label or "New Delhi")


@mock.patch.dict(os.environ, {"LLM_API_KEY": ""})
@mock.patch.object(services, "live_report", side_effect=fake_report)
class PageTests(TestCase):
    def setUp(self):
        cache.clear()

    def test_pages_load(self, _):
        for name in ("now", "check", "guide", "ask", "trends", "about", "model"):
            with self.subTest(page=name):
                self.assertEqual(self.client.get(reverse(name)).status_code, 200)

    def test_now_shows_live_report(self, _):
        res = self.client.get(reverse("now"))
        self.assertContains(res, "New Delhi")
        self.assertContains(res, "Best time to be outside")
        self.assertContains(res, 'data-advice="children"')
        self.assertContains(res, "GRAP")  # in Delhi-NCR

    def test_report_partial_for_a_place(self, live):
        res = self.client.post(reverse("report"), {"lat": 19.08, "lon": 72.88, "label": "Mumbai, Maharashtra"})
        self.assertContains(res, "Mumbai, Maharashtra")
        self.assertNotContains(res, "Delhi-NCR, GRAP")  # GRAP only applies near Delhi
        live.assert_called_with(19.08, 72.88, "Mumbai, Maharashtra")
        self.assertEqual(self.client.post(reverse("report"), {"lat": 200, "lon": 0}).status_code, 400)
        self.assertEqual(self.client.get(reverse("report")).status_code, 405)

    def test_live_failure_shows_friendly_error(self, live):
        from core.report import ReportError
        live.side_effect = ReportError("live data is unavailable right now (ConnectionError)")
        res = self.client.get(reverse("now"))
        self.assertContains(res, "Couldn't load live air quality")
        self.assertContains(self.client.post(reverse("report"), {"lat": 19, "lon": 72}), "Try again", status_code=502)

    def test_example_fills_form(self, _):
        p = services.predictor().presets[1]
        res = self.client.get(reverse("check"), {"example": p["key"]})
        self.assertContains(res, f'value="{p["values"]["pm10"]}"')

    def test_valid_post_shows_result_and_saves(self, _):
        res = self.client.post(reverse("check"), {**sample_values(), "profile": "children"})
        self.assertContains(res, "Your readings")
        self.assertContains(res, "Remain indoors and keep activity levels low.")
        rec = Prediction.objects.get()
        self.assertEqual(rec.source, Prediction.Source.WEB)
        self.assertGreater(rec.aqi, 0)

    def test_negative_value_is_rejected(self, _):
        res = self.client.post(reverse("check"), {**sample_values(), "co": -5, "profile": "general"})
        self.assertContains(res, "Ensure this value is greater than or equal to 0")
        self.assertEqual(Prediction.objects.count(), 0)

    def test_ozone_driven_result_shows_8_hour_note(self, _):
        res = self.client.post(reverse("check"), {**sample_values(), "o3": 400, "pm10": 30, "co": 300, "profile": "general"})
        self.assertContains(res, "8-hour average")

    def test_missing_tables_do_not_crash(self, _):
        with mock.patch.object(Prediction.objects, "create", side_effect=OperationalError("no such table: predictor_prediction")):
            res = self.client.post(reverse("check"), {**sample_values(), "profile": "general"})
        self.assertContains(res, "python manage.py migrate")

    def test_guide_page(self, _):
        res = self.client.get(reverse("guide"), {"category": "Poor"})
        self.assertContains(res, "Avoid outdoor physical exertion.")
        self.assertContains(res, 'level chosen')
        self.assertContains(res, "Stage IV")

    def test_ask_offline_answer_has_sources(self, _):
        res = self.client.get(reverse("ask"), {"q": "Should I wear an N95 mask?"})
        self.assertContains(res, "Health Ministry")

    def test_user_pages_hide_developer_details(self, _):
        for name in ("now", "check", "ask"):
            body = self.client.get(reverse(name)).content.decode()
            for word in ("Gradient boosting", "/api/v1/predict", "offline mode", "LLM"):
                self.assertNotIn(word, body, f"{word!r} shown on {name}")

    def test_mobile_shell_and_pwa_bits(self, _):
        res = self.client.get(reverse("now"))
        self.assertContains(res, 'class="tabbar"')
        self.assertContains(res, 'viewport-fit=cover')
        self.assertContains(res, "manifest.webmanifest")
        self.assertContains(res, 'id="theme-toggle"')

    def test_old_addresses_redirect(self, _):
        for old, new in (("/dashboard/", "/trends/"), ("/measures/?category=Poor", "/what-to-do/?category=Poor"),
                         ("/assistant/", "/ask/"), ("/model/", "/about/model/")):
            with self.subTest(old=old):
                self.assertRedirects(self.client.get(old), new, status_code=301)

    def test_history_is_staff_only(self, _):
        self.assertEqual(self.client.get(reverse("history")).status_code, 302)  # to the admin login
        from django.contrib.auth.models import User
        User.objects.create_superuser("admin", "a@example.com", "pass12345")
        self.client.login(username="admin", password="pass12345")
        services.run_prediction(sample_values(), explain=False)
        self.assertContains(self.client.get(reverse("history")), "Website")
        self.assertEqual(self.client.get("/admin/predictor/prediction/").status_code, 200)

    def test_404_page(self, _):
        self.assertContains(self.client.get("/no-such-page/"), "Page not found", status_code=404)


@mock.patch.dict(os.environ, {"LLM_API_KEY": ""})
class ApiTests(TestCase):
    def setUp(self):
        cache.clear()
        self.api = APIClient()

    def test_predict(self):
        res = self.api.post("/api/v1/predict", {**sample_values(), "explain": True}, format="json")
        self.assertEqual(res.status_code, 200, res.content)
        body = res.json()
        self.assertIn("pm25", body)
        self.assertIn(body["aqi"]["category"]["name"], ml.CATEGORY_ORDER)
        self.assertEqual(body["explanation"]["mode"], "template")
        self.assertEqual(Prediction.objects.get().source, Prediction.Source.API)

    def test_predict_validation(self):
        bad = {**sample_values(), "pm10": -1}
        res = self.api.post("/api/v1/predict", bad, format="json")
        self.assertEqual(res.status_code, 400)
        self.assertIn("pm10", res.json())
        missing = sample_values()
        del missing["co"]
        self.assertEqual(self.api.post("/api/v1/predict", missing, format="json").status_code, 400)

    def test_history_is_paginated_and_staff_only(self):
        for _ in range(3):
            services.run_prediction(sample_values(), explain=False)
        self.assertIn(self.api.get("/api/v1/history").status_code, (401, 403))
        from django.contrib.auth.models import User
        self.api.force_authenticate(User.objects.create_superuser("admin", "a@example.com", "pass12345"))
        body = self.api.get("/api/v1/history").json()
        self.assertEqual(body["count"], 3)
        self.assertEqual(len(body["results"]), 3)

    def test_stats(self):
        self.assertIn("month_of_year", self.api.get("/api/v1/stats").json())
        self.assertEqual(self.api.get("/api/v1/stats", {"year": 2021, "month": 11}).json()["month"], "Nov")
        self.assertEqual(self.api.get("/api/v1/stats", {"month": 13}).status_code, 400)
        self.assertEqual(self.api.get("/api/v1/stats", {"year": 2015}).status_code, 404)

    def test_ask(self):
        body = self.api.post("/api/v1/ask", {"question": "What happens under GRAP Stage IV?"}, format="json").json()
        self.assertEqual(body["mode"], "offline")
        self.assertTrue(body["citations"][0]["id"].startswith("grap_stages#"))

    def test_live_endpoint(self):
        fake = {"source": "open-meteo", "values": sample_values(), "estimated": ["nh3"]}
        with mock.patch.object(services, "live_readings", return_value=fake) as m:
            self.assertEqual(self.api.get("/api/v1/live").json()["source"], "open-meteo")
            m.assert_called_with()
            res = self.api.post("/api/v1/live", {"lat": 19.076, "lon": 72.8777}, format="json")
            self.assertEqual(res.status_code, 200)
            m.assert_called_with(19.076, 72.8777)

    def test_live_location_validation(self):
        self.assertEqual(self.api.post("/api/v1/live", {"lat": 120, "lon": 10}, format="json").status_code, 400)
        self.assertEqual(self.api.post("/api/v1/live", {"lat": 10}, format="json").status_code, 400)

    def test_live_location_end_to_end_offline(self):
        from core import live
        live.clear_cache()
        import requests
        with mock.patch("core.live.requests.get", side_effect=requests.ConnectionError("offline")):
            body = self.api.post("/api/v1/live", {"lat": 19.076, "lon": 72.8777}, format="json").json()
        self.assertEqual(body["source"], "fallback")
        self.assertFalse(body["place"]["near_delhi"])
        self.assertGreater(body["place"]["distance_from_delhi_km"], 1000)

    def test_report_api(self):
        with mock.patch.object(services, "live_report", side_effect=fake_report) as m:
            self.assertEqual(self.api.get("/api/v1/report").json()["place"]["label"], "New Delhi")
            body = self.api.post("/api/v1/report", {"lat": 12.97, "lon": 77.59, "label": "Bengaluru"}, format="json").json()
            self.assertEqual(body["place"]["label"], "Bengaluru")
            self.assertIn("best_window", body)
        self.assertEqual(self.api.post("/api/v1/report", {"lat": 95, "lon": 0}, format="json").status_code, 400)
        from core.report import ReportError
        with mock.patch.object(services, "live_report", side_effect=ReportError("down")):
            self.assertEqual(self.api.get("/api/v1/report").status_code, 502)

    def test_places_api(self):
        with mock.patch.object(services, "search_places", return_value=[{"label": "Pune, Maharashtra, India"}]):
            self.assertEqual(self.api.get("/api/v1/places", {"q": "Pune"}).json()["results"][0]["label"], "Pune, Maharashtra, India")

    def test_measures_api(self):
        self.assertEqual(len(self.api.get("/api/v1/measures").json()["categories"]), 6)
        body = self.api.get("/api/v1/measures", {"aqi": 320, "profile": "elderly"}).json()
        self.assertEqual(body["category"], "Very Poor")
        self.assertTrue(body["you_are_vulnerable"])
        self.assertEqual(body["grap"][0]["stage"], "Stage II")
        self.assertEqual(self.api.get("/api/v1/measures", {"aqi": 900}).status_code, 400)
        self.assertEqual(self.api.get("/api/v1/measures", {"profile": "cat"}).status_code, 400)

    def test_predict_includes_measures(self):
        body = self.api.post("/api/v1/predict", sample_values(), format="json").json()
        self.assertEqual(body["measures"]["category"], body["aqi"]["category"]["name"])

    def test_schema_and_docs(self):
        self.assertEqual(self.api.get("/api/schema/").status_code, 200)
        self.assertEqual(self.client.get("/api/docs/").status_code, 200)
