"""Tests for the framework-independent core. Run: python -m unittest discover tests_core"""
import json
import unittest

from core import assistant, live, measures, ml, naqi, rag, report, stats
from core.llm import LLMClient, LLMError


class NaqiTests(unittest.TestCase):
    def test_band_edges(self):
        self.assertEqual(naqi.sub_index("pm2_5", 0), 0)
        self.assertEqual(naqi.sub_index("pm2_5", 30), 50)
        self.assertEqual(naqi.sub_index("pm2_5", 60), 100)
        self.assertEqual(naqi.sub_index("pm2_5", 90), 200)
        self.assertEqual(naqi.sub_index("pm2_5", 250), 400)

    def test_gap_and_cap(self):
        self.assertEqual(naqi.sub_index("pm2_5", 30.5), 51)  # value in the 30-31 gap starts the next band
        self.assertEqual(naqi.sub_index("pm2_5", 5000), 500)

    def test_interpolation(self):
        # PM10 175 is the middle of 101-250 (Moderate, AQI 101-200)
        self.assertAlmostEqual(naqi.sub_index("pm10", 175.5), 150.5, delta=0.5)

    def test_overall_takes_worst_and_converts_co(self):
        r = naqi.overall_aqi({"pm2_5": 100, "pm10": 80, "no2": 30, "co": 2000})  # CO given in µg/m³
        self.assertEqual(r["prominent"], "pm2_5")
        self.assertEqual(r["category"]["name"], "Poor")
        self.assertEqual(r["sub_indices"]["co"], 100)  # 2 mg/m³ is the top of Satisfactory
        self.assertEqual(r["grap"]["stage"], "Stage I")

    def test_needs_three_pollutants_with_pm(self):
        with self.assertRaises(ValueError):
            naqi.overall_aqi({"no2": 10, "so2": 10, "co": 500})

    def test_grap_stages(self):
        self.assertIsNone(naqi.grap_stage(150))
        self.assertEqual(naqi.grap_stage(420)["stage"], "Stage III")
        self.assertEqual(naqi.grap_stage(480)["stage"], "Stage IV")


class ModelLoadTests(unittest.TestCase):
    def test_version_mismatch_refits_quickly(self):
        import joblib, shutil, tempfile
        from pathlib import Path
        tmp = Path(tempfile.mkdtemp())
        for f in ("model.joblib", "model_meta.json"):
            shutil.copy(ml.MODEL_PATH.parent / f, tmp / f)
        saved = joblib.load(tmp / "model.joblib")
        saved["sklearn_version"] = "0.0.0"
        joblib.dump(saved, tmp / "model.joblib")
        logs = []
        p = ml.Predictor.load(tmp / "model.joblib", log=logs.append)
        self.assertTrue(any("Refitted" in m for m in logs))
        self.assertEqual(joblib.load(tmp / "model.joblib")["sklearn_version"], __import__("sklearn").__version__)
        self.assertGreater(p.predict(p.presets[0]["values"]).pm25, 0)
        shutil.rmtree(tmp)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = ml.Predictor.load()
        cls.metrics = ml.load_metrics()

    def test_metrics_beat_baseline(self):
        m = self.metrics["models"]
        self.assertLess(m[self.metrics["best_model"]]["mae"], m["PM10 ratio (baseline)"]["mae"])
        self.assertGreater(m[self.metrics["best_model"]]["category_accuracy"], 0.85)

    def test_test_period_is_after_training(self):
        self.assertLess(self.metrics["train"]["to"], self.metrics["test"]["from"])

    def test_prediction_is_consistent(self):
        r = self.p.predict(self.p.presets[0]["values"])
        self.assertEqual(r.pm25_category, naqi.pm25_category(r.pm25))
        self.assertGreaterEqual(r.aqi["aqi"], naqi.sub_index("pm2_5", r.pm25) - 1)
        self.assertEqual(len(r.contributions), len(ml.FEATURES))

    def test_preset_close_to_reality(self):
        for preset in self.p.presets:
            r = self.p.predict(preset["values"])
            self.assertLess(abs(r.pm25 - preset["actual_pm25_median"]), 0.25 * preset["actual_pm25_median"] + 15)

    def test_rejects_bad_input(self):
        vals = dict(self.p.presets[0]["values"])
        vals["co"] = -1
        with self.assertRaises(ValueError):
            self.p.predict(vals)
        del vals["co"]
        with self.assertRaises(ValueError):
            self.p.predict(vals)

    def test_diwali_preset_uses_only_diwali_weeks(self):
        d = next(x for x in self.p.presets if x["key"] == "diwali_week")
        self.assertEqual(d["hours"], 14 * 4)  # 2 windows x 7 days x 4 evening hours
        r = self.p.predict(d["values"])
        self.assertLess(abs(r.pm25 - d["actual_pm25_median"]), 0.1 * d["actual_pm25_median"])

    def test_out_of_range_warning(self):
        vals = dict(self.p.presets[0]["values"])
        vals["pm10"] = 99999
        self.assertTrue(any("PM10" in w for w in self.p.predict(vals).warnings))


class MeasuresTests(unittest.TestCase):
    def test_every_category_has_measures(self):
        rows = measures.table()
        self.assertEqual([r["category"] for r in rows], [c[0] for c in naqi.CATEGORIES])
        self.assertTrue(all(r["general_public"] and r["vulnerable_groups"] for r in rows))

    def test_profile_switches_advice(self):
        self.assertEqual(measures.for_aqi(350, "general")["you"], measures.PERSONAL["Very Poor"][0])
        self.assertEqual(measures.for_aqi(350, "children")["you"], "Remain indoors and keep activity levels low.")

    def test_grap_matches_aqi(self):
        self.assertEqual(measures.for_aqi(150)["grap"], [])
        self.assertEqual([g["stage"] for g in measures.for_aqi(250)["grap"]], ["Stage I"])
        self.assertEqual([g["stage"] for g in measures.for_aqi(470)["grap"]], ["Stage IV"])
        self.assertEqual([g["stage"] for g in measures.table()[-1]["grap"]], ["Stage III", "Stage IV"])

    def test_steps_grow_with_level(self):
        self.assertEqual(measures.for_aqi(30)["home_steps"], [])
        self.assertLess(len(measures.for_aqi(150)["home_steps"]), len(measures.for_aqi(250)["home_steps"]))

    def test_assistant_answers_measures_question(self):
        r = assistant.ask("What measures should I take when the AQI is very poor?", client=LLMClient(api_key=""))
        self.assertEqual(r["citations"][0]["id"], "aqi_actions#what-to-do-when-the-aqi-is-very-poor")

    def test_topic_gate(self):
        self.assertFalse(rag.on_topic("Suggest a good movie to watch tonight"))
        self.assertTrue(rag.on_topic("What should I do when the AQI is good?"))


class ReportTests(unittest.TestCase):
    def test_cpcb_style_averages(self):
        from datetime import datetime, timedelta
        hours = 72
        t0 = datetime(2026, 10, 7)
        times = [(t0 + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(hours)]
        # PM2.5: 0 for the last 24 h except the newest hour; ozone high only in the last 8 h
        pm25 = [500.0] * 6 + [24.0] * (hours - 6)
        o3 = [0.0] * 22 + [160.0] * 8 + [0.0] * (hours - 30)
        payload = {"current": {"time": times[29]}, "hourly": {"time": times, "pm2_5": pm25, "pm10": [40.0] * hours,
                   "nitrogen_dioxide": [20.0] * hours, "sulphur_dioxide": [5.0] * hours, "ozone": o3,
                   "carbon_monoxide": [500.0] * hours, "ammonia": [None] * hours}}
        rep = report.build_report(payload, 28.61, 77.21, "Delhi")
        subs = rep["aqi"]["sub_indices"]
        self.assertEqual(subs["pm2_5"], round(naqi.sub_index("pm2_5", 24.0)))   # 24-h mean ignores the old spike
        self.assertEqual(subs["o3"], round(naqi.sub_index("o3", 160.0)))          # 8-h mean of the last 8 hours
        self.assertNotIn("nh3", subs)                                               # missing, not guessed
        self.assertEqual(rep["missing"], ["NH₃"])
        self.assertEqual(len(rep["outlook"]), 24)
        self.assertTrue(rep["place"]["in_delhi_ncr"])

    def test_best_window_is_daytime_and_cleanest(self):
        from datetime import datetime, timedelta
        hours = 72
        t0 = datetime(2026, 10, 7)
        times = [(t0 + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(hours)]
        pm25 = [200.0] * hours
        pm25[24 + 14] = pm25[24 + 15] = 20.0   # clean at 2-4 pm tomorrow
        pm25[24 + 3] = 5.0                       # cleaner at 3 am, but that's night
        payload = {"current": {"time": times[23]}, "hourly": {"time": times, "pm2_5": pm25, "pm10": [30.0] * hours,
                   "nitrogen_dioxide": [20.0] * hours, "sulphur_dioxide": [5.0] * hours, "ozone": [30.0] * hours,
                   "carbon_monoxide": [500.0] * hours, "ammonia": [None] * hours}}
        best = report.build_report(payload, 19.0, 72.8, "Mumbai")["best_window"]
        self.assertEqual((best["from"], best["to"]), ("2 pm", "4 pm"))

    def test_not_enough_data_raises(self):
        from datetime import datetime, timedelta
        times = [(datetime(2026, 10, 7) + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(48)]
        empty = [None] * 48
        payload = {"current": {"time": times[30]}, "hourly": {"time": times, **{k: empty for k in report.HOURLY}}}
        with self.assertRaises(report.ReportError):
            report.build_report(payload, 28.6, 77.2, "x")

    def test_fetch_errors_become_report_errors_and_cache(self):
        import requests
        report.clear_cache()

        def boom(*a, **k):
            raise requests.ConnectionError("down")
        with self.assertRaises(report.ReportError):
            report.fetch_report(get=boom)

    def test_search_places_puts_india_first(self):
        report.clear_cache()
        rows = {"results": [{"name": "Pune", "admin1": "Pará", "country": "Brazil", "country_code": "BR", "latitude": 1, "longitude": 2},
                            {"name": "Pune", "admin1": "Maharashtra", "country": "India", "country_code": "IN", "latitude": 18.5, "longitude": 73.8}]}
        out = report.search_places("Pune", get=lambda *a, **k: FakeResponse(rows))
        self.assertEqual(out[0]["label"], "Pune, Maharashtra, India")
        self.assertEqual(report.search_places("x"), [])


class StatsTests(unittest.TestCase):
    def test_dashboard_shapes(self):
        d = stats.dashboard()
        self.assertEqual(len(d["month_of_year"]["values"]), 12)
        self.assertEqual(len(d["hour_of_day"]["all"]), 24)
        self.assertAlmostEqual(sum(d["category_share"].values()), 1.0, delta=0.01)

    def test_tools_handle_missing_periods(self):
        self.assertIn("error", stats.year_summary(2019))
        self.assertEqual(stats.year_summary(2020)["note"], "Partial year")
        self.assertEqual(len(stats.worst_days(3)["days"]), 3)


class RagTests(unittest.TestCase):
    def test_every_chunk_has_a_source(self):
        for c in rag.load_chunks():
            self.assertTrue(c.url and c.source and c.text, c.id)

    def test_evaluation_quality(self):
        r = rag.evaluate()
        self.assertGreaterEqual(r["hit_at_3"], 0.85)
        self.assertGreaterEqual(r["refusal_rate"], 0.8)

    def test_out_of_scope_returns_nothing(self):
        self.assertEqual(rag.retrieve("best recipe for butter chicken"), [])

    def test_air_cleaner_section_and_everyday_verbs(self):
        self.assertEqual(rag.retrieve("Do air purifiers help?")[0]["id"], "air_cleaners#do-air-purifiers-help-with-indoor-pollution")
        self.assertEqual(rag.retrieve("Suggest a good movie to watch tonight"), [])

    def test_stage_numbers(self):
        self.assertTrue(rag.retrieve("stage 4 rules")[0]["id"].startswith("grap_stages#grap-stage-iv"))


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(str(self.status_code))


def fake_llm(script):
    """A fake OpenAI-compatible endpoint that returns the scripted messages in order."""
    calls = []

    def post(url, json=None, timeout=None, headers=None):
        calls.append(json)
        msg = script[len(calls) - 1]
        return FakeResponse({"choices": [{"message": msg}]})
    return LLMClient(api_key="test", post=post), calls


class AssistantTests(unittest.TestCase):
    def test_offline_guidance_cites_source(self):
        r = assistant.ask("Should I wear a cloth mask?", client=LLMClient(api_key=""))
        self.assertEqual(r["mode"], "offline")
        self.assertEqual(r["citations"][0]["id"], "health_advisory#masks-and-n95-respirators")

    def test_offline_refuses_out_of_scope(self):
        r = assistant.ask("Who won the cricket world cup?", client=LLMClient(api_key=""))
        self.assertIn("don't have", r["answer"])

    def test_offline_data_question(self):
        r = assistant.ask("Which month was worst in 2021?", client=LLMClient(api_key=""))
        self.assertEqual(r["steps"][0]["tool"], "year_summary")
        self.assertIn(stats.year_summary(2021)["worst_month"]["month"], r["answer"])

    def test_llm_tool_loop_and_citations(self):
        client, calls = fake_llm([
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "search_guidelines", "arguments": json.dumps({"query": "N95 mask"})}}]},
            {"role": "assistant", "content": "A well-fitted N95 can help for short exposure; cloth masks are not effective [1]."},
        ])
        r = assistant.ask("Do masks work?", client=client)
        self.assertEqual(r["mode"], "llm")
        self.assertEqual(r["citations"][0]["id"], "health_advisory#masks-and-n95-respirators")
        self.assertNotIn("warning", r)
        # the tool result was sent back to the model
        self.assertEqual(calls[1]["messages"][-1]["role"], "tool")

    def test_llm_invented_number_is_flagged(self):
        client, _ = fake_llm([
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "year_summary", "arguments": "{\"year\": 2021}"}}]},
            {"role": "assistant", "content": "The average in 2021 was 987 µg/m³."},
        ])
        r = assistant.ask("How bad was 2021?", client=client)
        self.assertIn("warning", r)

    def test_llm_failure_falls_back(self):
        def post(*a, **k):
            return FakeResponse({"error": {"message": "rate limited"}}, status=429)
        r = assistant.ask("Should I wear an N95?", client=LLMClient(api_key="x", post=post))
        self.assertEqual(r["mode"], "offline")
        self.assertIn("429", r["notice"])

    def test_explain_guard(self):
        p = ml.Predictor.load()
        pred = p.predict(p.presets[0]["values"]).as_dict()
        good = f"The AQI is {pred['aqi']['aqi']}, which is {pred['aqi']['category']['name']}. Avoid outdoor exercise [1]."
        client, _ = fake_llm([{"role": "assistant", "content": good}])
        self.assertEqual(assistant.explain(pred, "general", client)["mode"], "llm")
        client, _ = fake_llm([{"role": "assistant", "content": "The AQI is 42, which is Good."}])
        r = assistant.explain(pred, "general", client)
        self.assertEqual(r["mode"], "template")
        self.assertIn("accuracy check", r["notice"])

    def test_template_without_key(self):
        p = ml.Predictor.load()
        pred = p.predict(p.presets[-1]["values"]).as_dict()
        r = assistant.explain(pred, "children", LLMClient(api_key=""))
        self.assertEqual(r["mode"], "template")
        self.assertIn(str(pred["aqi"]["aqi"]), r["text"])

    def test_number_guard(self):
        self.assertEqual(assistant.unsupported_numbers("PM2.5 was 238 [2]", ["mean 238.1"]), [])
        self.assertEqual(assistant.unsupported_numbers("PM2.5 was 512", ["mean 238.1"]), [512.0])


class LiveTests(unittest.TestCase):
    def setUp(self):
        live.clear_cache()
        self.medians = {f: 1.0 for f in ml.FEATURES}

    def test_success_maps_fields_and_marks_estimates(self):
        cur = {"time": "2026-10-07T12:00", "carbon_monoxide": 900, "nitrogen_monoxide": 2, "nitrogen_dioxide": 30,
               "ozone": 80, "sulphur_dioxide": 12, "pm10": 140, "ammonia": None, "pm2_5": 70}
        r = live.fetch_current(self.medians, {}, get=lambda *a, **k: FakeResponse({"current": cur}))
        self.assertEqual(r["source"], "open-meteo")
        self.assertEqual(r["values"]["co"], 900)
        self.assertEqual(r["estimated"], ["nh3"])
        self.assertEqual(r["reported_pm25"], 70)
        # second call is served from cache
        r2 = live.fetch_current(self.medians, {}, get=lambda *a, **k: self.fail("should be cached"))
        self.assertTrue(r2["cached"])

    def test_location_is_cached_separately_and_distance(self):
        calls = []

        def get(url, params=None, timeout=None):
            calls.append(params)
            return FakeResponse({"current": {"time": "t", "pm10": 50, "carbon_monoxide": 300}})
        a = live.fetch_current(self.medians, {}, get=get)
        b = live.fetch_current(self.medians, {}, lat=19.076, lon=72.8777, get=get)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["latitude"], 19.08)
        self.assertTrue(a["place"]["near_delhi"])
        self.assertAlmostEqual(b["place"]["distance_from_delhi_km"], 1150, delta=30)  # Delhi to Mumbai
        self.assertIn("nh3", b["estimated"])
        with self.assertRaises(ValueError):
            live.fetch_current(self.medians, {}, lat=95, lon=0, get=get)

    def test_failure_falls_back(self):
        import requests

        def boom(*a, **k):
            raise requests.ConnectionError("down")
        r = live.fetch_current(self.medians, {"co": 5}, get=boom)
        self.assertEqual(r["source"], "fallback")
        self.assertIn("unavailable", r["error"])


if __name__ == "__main__":
    unittest.main()
