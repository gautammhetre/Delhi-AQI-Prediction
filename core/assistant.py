"""
The AQI assistant: RAG + tool calling, with an offline fallback.

Division of labour (the rule that keeps it honest):
  * the ML model and the NAQI maths produce every number about a prediction,
  * the data tools produce every number about Delhi's history,
  * retrieval supplies the official guidance, with sources,
  * the LLM only chooses tools and writes the answer from what they returned.

A guard checks the LLM's answer: numbers it states must appear in the evidence it was
given, and an explanation must name the category the model predicted. If the check
fails, explanations fall back to a template and answers carry a warning.
"""
from __future__ import annotations

import json
import re

from . import naqi, rag, stats
from .llm import LLMClient, LLMError

MONTH_NAMES = {m.lower(): i + 1 for i, m in enumerate(stats.MONTHS)}
MONTH_NAMES.update({"january": 1, "february": 2, "march": 3, "april": 4, "june": 6, "july": 7, "august": 8,
                    "september": 9, "october": 10, "november": 11, "december": 12, "sept": 9})

PROFILES = {
    "general": "a healthy adult",
    "children": "a parent of young children",
    "elderly": "an older adult",
    "lung_heart": "someone with asthma, lung or heart disease",
    "pregnant": "a pregnant woman",
    "outdoor_worker": "someone who works outdoors",
}

# ------------------------------------------------------------------ tools

TOOLS = [
    {"type": "function", "function": {
        "name": "search_guidelines",
        "description": "Search official guidance (CPCB AQI categories and breakpoints, GRAP stages for Delhi-NCR, "
                       "WHO guidelines, the Health Ministry advisory on precautions, masks, exercise, symptoms) "
                       "and notes on this app. Returns numbered passages with sources.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "monthly_average",
        "description": "Average PM2.5 in Delhi for a calendar month, optionally for one year (data covers Nov 2020 to Jan 2023).",
        "parameters": {"type": "object", "properties": {
            "year": {"type": "integer"}, "month": {"type": "integer", "minimum": 1, "maximum": 12}}}}},
    {"type": "function", "function": {
        "name": "worst_days",
        "description": "Days with the highest daily-average PM2.5, optionally within one year.",
        "parameters": {"type": "object", "properties": {
            "n": {"type": "integer", "minimum": 1, "maximum": 20}, "year": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "hourly_profile",
        "description": "Average PM2.5 by hour of day (IST), optionally for one calendar month.",
        "parameters": {"type": "object", "properties": {"month": {"type": "integer", "minimum": 1, "maximum": 12}}}}},
    {"type": "function", "function": {
        "name": "year_summary",
        "description": "A year's mean PM2.5, worst and best month, and share of Severe hours.",
        "parameters": {"type": "object", "properties": {"year": {"type": "integer"}}, "required": ["year"]}}},
]

DATA_TOOLS = {
    "monthly_average": stats.monthly_average,
    "worst_days": stats.worst_days,
    "hourly_profile": stats.hourly_profile,
    "year_summary": stats.year_summary,
}

SYSTEM_PROMPT = """You are the assistant inside a Delhi air-quality app.
Answer only from tool results. Use search_guidelines for health advice, AQI meanings, GRAP and WHO questions,
and the data tools for questions about Delhi's historical pollution.
Rules:
- Every factual sentence from search_guidelines must cite its passage number like [1].
- Only state numbers that appear in tool results. Never estimate or invent numbers.
- If the tools return nothing relevant, say you don't have that information. Do not use outside knowledge.
- This is not medical advice; for symptoms, point to the doctor guidance if it was retrieved.
- Plain language, under 150 words."""


class _Session:
    """Collects the evidence a single answer is allowed to use."""

    def __init__(self):
        self.sources: list[dict] = []
        self.evidence: list[str] = []
        self.steps: list[dict] = []

    def run(self, name: str, args: dict) -> dict:
        if name == "search_guidelines":
            hits = rag.retrieve(str(args.get("query", "")), k=3)
            out = []
            for h in hits:
                if h["id"] not in [s["id"] for s in self.sources]:
                    self.sources.append(h)
                ref = [s["id"] for s in self.sources].index(h["id"]) + 1
                out.append({"ref": ref, "title": h["title"], "section": h["section"], "text": h["text"]})
                self.evidence.append(h["text"])
            result = {"passages": out} if out else {"passages": [], "note": "Nothing relevant in the knowledge base."}
        elif name in DATA_TOOLS:
            clean = {k: v for k, v in args.items() if v is not None}
            try:
                result = DATA_TOOLS[name](**clean)
            except (TypeError, ValueError) as e:
                result = {"error": f"Bad arguments: {e}"}
            self.evidence.append(json.dumps(result))
        else:
            result = {"error": f"Unknown tool {name}"}
        self.steps.append({"tool": name, "args": args, "summary": _summarise(name, result)})
        return result


def _summarise(name, result):
    if "error" in result:
        return result["error"]
    if name == "search_guidelines":
        return f"{len(result['passages'])} passages"
    if name == "worst_days":
        return f"{len(result['days'])} days"
    return "ok"


# ------------------------------------------------------------------ guard

_NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\w])")


def _numbers(text: str) -> set[float]:
    out = set()
    for m in _NUM.findall(text or ""):
        try:
            out.add(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def unsupported_numbers(answer: str, evidence: list[str], extra: str = "") -> list[float]:
    """Numbers in the answer that don't appear in the evidence (allowing rounding).
    Citation markers like [2] and small counting numbers (0-10) are ignored."""
    stripped = re.sub(r"\[\d+\]", "", answer or "")
    allowed = set()
    for e in evidence + [extra]:
        allowed |= _numbers(e)
    bad = []
    for n in _numbers(stripped):
        if n <= 10 and n == int(n):
            continue
        if any(abs(n - a) <= max(1.0, 0.01 * a) for a in allowed):
            continue
        bad.append(n)
    return sorted(bad)


# ------------------------------------------------------------------ ask()

def ask(question: str, client: LLMClient | None = None) -> dict:
    question = (question or "").strip()[:500]
    if not question:
        return {"mode": "offline", "answer": "Please type a question.", "citations": [], "steps": []}
    client = client or LLMClient()
    if client.is_configured:
        try:
            return _ask_llm(question, client)
        except LLMError as e:
            result = _ask_offline(question)
            result["notice"] = f"The language model was unavailable ({e}), so this answer was built without it."
            return result
    result = _ask_offline(question)
    result["notice"] = "No LLM key is configured, so this answer comes straight from the sources and data tools."
    return result


def _ask_llm(question: str, client: LLMClient, max_turns: int = 5) -> dict:
    s = _Session()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    for _ in range(max_turns):
        msg = client.chat(messages, tools=TOOLS)
        calls = msg.get("tool_calls") or []
        if not calls:
            answer = (msg.get("content") or "").strip()
            cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 0 < int(n) <= len(s.sources)})
            bad = unsupported_numbers(answer, s.evidence, question)
            out = {"mode": "llm", "model": client.model, "answer": answer, "steps": s.steps,
                   "citations": [{**s.sources[i - 1], "ref": i} for i in cited],
                   "consulted": [{**src, "ref": i + 1} for i, src in enumerate(s.sources)]}
            if bad:
                out["warning"] = ("Some numbers in this answer were not found in the sources it used "
                                  f"({', '.join(f'{n:g}' for n in bad)}). Check them before relying on them.")
            return out
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for c in calls:
            fn = c.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            result = s.run(fn.get("name", ""), args if isinstance(args, dict) else {})
            messages.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": json.dumps(result)})
    raise LLMError("the model did not finish within the turn limit")


def _parse_period(q: str):
    ql = q.lower()
    year = next((int(y) for y in re.findall(r"\b(20[12]\d)\b", ql)), None)
    month = next((MONTH_NAMES[w] for w in re.findall(r"[a-z]+", ql) if w in MONTH_NAMES and w != "may"), None)
    if month is None and re.search(r"\bmay\b", ql) and re.search(r"\bin may\b|\bmay 20", ql):
        month = 5
    return year, month


def _ask_offline(question: str) -> dict:
    s = _Session()
    ql = question.lower()
    year, month = _parse_period(question)
    data_q = re.search(r"\b(worst|best|cleanest|dirtiest|highest|lowest|average|mean|how (bad|polluted))\b", ql) and \
        (year or month or re.search(r"\b(day|days|month|months|hour|time of day|year)\b", ql))

    if data_q:
        if re.search(r"\b(hour|time of day|what time)\b", ql):
            r = s.run("hourly_profile", {"month": month})
            text = (f"On average PM2.5 is highest around {r['worst_hour']} and lowest around {r['best_hour']} (IST)"
                    f"{' in ' + r['month'] if month else ''}.")
        elif re.search(r"\bdays?\b", ql):
            r = s.run("worst_days", {"n": 5, "year": year})
            if "error" in r:
                text = f"I don't have data for that year. The data covers {stats.coverage()['from']} to {stats.coverage()['to']}."
            else:
                rows = "; ".join(f"{d['date']} ({d['mean_pm25']:g} µg/m³)" for d in r["days"])
                text = f"Days with the highest average PM2.5{' in ' + str(year) if year else ''}: {rows}."
        elif year and not month:
            r = s.run("year_summary", {"year": year})
            text = (f"I don't have data for {year}." if "error" in r else
                    f"In {year} ({r['note'].lower()}, {', '.join(r['months_covered'][:1])}–{r['months_covered'][-1]}) the average PM2.5 was "
                    f"{r['mean_pm25']:g} µg/m³. The worst month was {r['worst_month']['month']} ({r['worst_month']['mean_pm25']:g}) and the "
                    f"cleanest was {r['best_month']['month']} ({r['best_month']['mean_pm25']:g}). {round(r['share_severe_hours'] * 100)}% of hours were Severe.")
        else:
            r = s.run("monthly_average", {"year": year, "month": month})
            if "error" in r:
                text = "I don't have data for that period."
            else:
                period = " ".join(str(x) for x in (r["month"], r["year"]) if x) or "the whole dataset"
                text = (f"Average PM2.5 in {period}: {r['mean_pm25']:g} µg/m³ ({r['pm25_category_of_mean']} on the NAQI PM2.5 scale), "
                        f"with {round(r['share_severe_hours'] * 100)}% of hours Severe.")
        return {"mode": "offline", "answer": text, "citations": [], "consulted": [], "steps": s.steps}

    r = s.run("search_guidelines", {"query": question})
    if not r["passages"]:
        return {"mode": "offline", "answer": "I don't have information about that. I can answer questions about Delhi's air quality, "
                "AQI categories, GRAP stages, WHO guidelines and health precautions.", "citations": [], "consulted": [], "steps": s.steps}
    best = r["passages"][0]
    answer = f"{best['text']} [1]"
    if len(r["passages"]) > 1:
        answer += f"\n\nAlso relevant: {r['passages'][1]['section']} [2]."
    cites = [{**src, "ref": i + 1} for i, src in enumerate(s.sources[:2])]
    return {"mode": "offline", "answer": answer, "citations": cites, "consulted": cites, "steps": s.steps}


# ------------------------------------------------------------------ explain a prediction

def _guidance_for(category_index: int, profile: str) -> list[dict]:
    queries = []
    if category_index >= 3:
        queries += ["outdoor exercise jogging polluted days", "opening windows ventilating home"]
        if profile in ("general", "outdoor_worker"):
            queries.append("masks n95 respirators")
    if profile != "general" and category_index >= 2:
        queries.append("who is most vulnerable to air pollution")
    if category_index >= 2:
        queries.append("symptoms to watch when to see a doctor")
    out, seen = [], set()
    for q in queries:
        for h in rag.retrieve(q, k=1):
            if h["id"] not in seen:
                seen.add(h["id"])
                out.append(h)
    return out


def explain_template(pred: dict, profile: str = "general") -> dict:
    aqi, cat = pred["aqi"], pred["aqi"]["category"]
    who = PROFILES.get(profile, PROFILES["general"])
    # Only mention drivers whose direction is intuitive: above-typical value that raised the estimate.
    drivers = [c for c in pred["contributions"] if c["effect"] > 0 and c["value"] > c["typical"]][:2]
    lines = [f"The estimated AQI is {aqi['aqi']} ({cat['name']}), driven mainly by {aqi['prominent_label']}. "
             f"CPCB's description for this level: {cat['health']}"]
    if drivers:
        lines.append("Compared with a typical hour in Delhi, " +
                     " and ".join(f"{d['label']} ({d['value']:g} vs a usual {d['typical']:g})" for d in drivers) +
                     " pushed the PM2.5 estimate up the most.")
    if aqi.get("grap"):
        g = aqi["grap"]
        lines.append(f"At this level Delhi-NCR's GRAP {g['stage']} measures would typically apply, if CAQM invokes them.")
    guidance = _guidance_for(cat["index"], profile)
    tips = {
        "health_advisory#outdoor-exercise-walking-and-jogging-on-polluted": "avoid outdoor exercise, especially early morning and late evening",
        "health_advisory#opening-windows-and-ventilating-the-home": "keep windows shut in the morning and evening and ventilate around midday",
        "health_advisory#masks-and-n95-respirators": "if you must be outside for a short time, a well-fitted N95 helps; cloth masks don't",
        "health_advisory#who-is-most-vulnerable-to-air-pollution": f"as {who}, you are in a group the Health Ministry lists as more vulnerable",
        "health_advisory#symptoms-to-watch-and-when-to-see-a-doctor": "see a doctor for breathlessness, chest pain, dizziness or persistent cough",
    }
    advice = [tips[g["id"]] for g in guidance if g["id"] in tips]
    if advice:
        lines.append(f"For {who}: " + "; ".join(advice) + ".")
    elif cat["index"] <= 1:
        lines.append("No special precautions are needed at this level.")
    return {"mode": "template", "text": " ".join(lines), "citations": [{**g, "ref": i + 1} for i, g in enumerate(guidance)]}


def explain(pred: dict, profile: str = "general", client: LLMClient | None = None) -> dict:
    client = client or LLMClient()
    if not client.is_configured:
        return explain_template(pred, profile)
    guidance = _guidance_for(pred["aqi"]["category"]["index"], profile)
    facts = {
        "aqi": pred["aqi"]["aqi"], "category": pred["aqi"]["category"]["name"],
        "cpcb_health_impact": pred["aqi"]["category"]["health"],
        "prominent_pollutant": pred["aqi"]["prominent_label"], "predicted_pm25_ugm3": pred["pm25"],
        "grap_stage_if_invoked": (pred["aqi"].get("grap") or {}).get("stage"),
        "top_drivers": [{k: c[k] for k in ("label", "value", "typical")} for c in pred["contributions"]
                        if c["effect"] > 0 and c["value"] > c["typical"]][:2],
    }
    passages = [{"ref": i + 1, "section": g["section"], "text": g["text"]} for i, g in enumerate(guidance)]
    prompt = (f"Explain this air-quality result to {PROFILES.get(profile, PROFILES['general'])} in 80-130 words. "
              f"Say the AQI and category exactly as given. Use only these facts and passages; cite passages like [1]. "
              f"No numbers that are not in the facts.\n\nFACTS: {json.dumps(facts)}\n\nPASSAGES: {json.dumps(passages)}")
    try:
        msg = client.chat([{"role": "system", "content": "You explain air-quality results plainly and accurately."},
                           {"role": "user", "content": prompt}], max_tokens=350)
    except LLMError as e:
        out = explain_template(pred, profile)
        out["notice"] = f"Language model unavailable ({e}); showing the standard explanation."
        return out
    text = (msg.get("content") or "").strip()
    bad = unsupported_numbers(text, [json.dumps(facts)] + [p["text"] for p in passages])
    if facts["category"].lower() not in text.lower() or bad:
        out = explain_template(pred, profile)
        out["notice"] = "The language model's explanation failed the accuracy check, so the standard explanation is shown."
        return out
    return {"mode": "llm", "model": client.model, "text": text, "citations": [{**g, "ref": i + 1} for i, g in enumerate(guidance)]}
