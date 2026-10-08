"""
What to do at each AQI level.

Personal measures come from the Health Ministry's advisory table "AQI levels, health effects and
certain protective health measures" (National Programme on Climate Change & Human Health, NCDC,
MoHFW, October 2023). Government measures are the CAQM's GRAP stages for Delhi-NCR.

Kept as data, not prose, so the website, the API and the assistant all show the same thing.
"""
from __future__ import annotations

from . import naqi

ADVISORY = {
    "title": "Health Advisory on Air Pollution",
    "source": "National Programme on Climate Change & Human Health, NCDC, Ministry of Health and Family Welfare (Oct 2023)",
    "url": "https://ncdc.mohfw.gov.in/wp-content/uploads/2024/07/Enclosure-Air-Pollution-Health-Advisory-Oct-2023.pdf",
}
GRAP_SOURCE = {
    "title": "Graded Response Action Plan (GRAP) for Delhi-NCR",
    "source": "Commission for Air Quality Management (CAQM); check caqm.nic.in for the schedule in force",
    "url": "https://caqm.nic.in/",
}

VULNERABLE_GROUPS = ("children (especially under 5), older adults, pregnant women, people with lung or heart "
                     "disease, and outdoor workers")

# category -> (general public, vulnerable groups), as in the MoHFW table
PERSONAL = {
    "Good": ("No special precautions.", "No special precautions."),
    "Satisfactory": ("No special precautions.", "Do less prolonged or strenuous outdoor physical exertion."),
    "Moderate": ("Do less prolonged or strenuous outdoor physical exertion.",
                 "Avoid prolonged or strenuous outdoor physical exertion."),
    "Poor": ("Avoid outdoor physical exertion.", "Avoid outdoor physical activities."),
    "Very Poor": ("Avoid outdoor physical activities, especially in the early morning and late evening.",
                  "Remain indoors and keep activity levels low."),
    "Severe": ("Avoid outdoor physical activities.", "Remain indoors and keep activity levels low."),
}

# Practical steps from the same advisory, shown from the level where they start to matter.
HOME_STEPS = [
    (3, "Keep doors and windows shut in the early morning and late evening; ventilate around midday (12–4 pm) if needed."),
    (3, "Wet-mop instead of sweeping; don't burn wood, waste, incense or firecrackers; don't smoke indoors."),
    (3, "If you must go out for a short time, a well-fitted N95 or N99 helps. Cloth, paper masks and scarves don't."),
    (2, "Check the AQI before planning outdoor activity, and avoid busy roads, construction and industrial areas."),
    (2, "See a doctor for breathlessness, chest pain, dizziness, persistent cough or red, watery eyes."),
]

# GRAP stage -> headline measures (each stage also keeps the measures of the stages below it)
GOVERNMENT = {
    "Stage I": ["Construction and demolition stopped on larger sites not registered for dust monitoring",
                "Vehicle pollution checks; action against visibly polluting vehicles and industries"],
    "Stage II": ["No coal or firewood in hotels, restaurants and eateries",
                 "Diesel generators restricted to emergency and essential services",
                 "Higher parking fees and more frequent public transport"],
    "Stage III": ["Wider ban on construction and demolition (essential projects excepted)",
                  "Stone crushers, brick kilns and mining closed",
                  "Possible restrictions on older BS III petrol and BS IV diesel cars"],
    "Stage IV": ["Most trucks barred from entering Delhi (essential goods and clean fuels excepted)",
                 "Most diesel vehicles barred except BS VI and essential services",
                 "States may move schools online, allow 50% work from home, or consider odd-even"],
}


def for_category(name: str, profile: str = "general") -> dict:
    """Measures for one NAQI category. `profile` other than "general" means a vulnerable group."""
    if name not in PERSONAL:
        raise ValueError(f"unknown category {name!r}")
    idx = [c[0] for c in naqi.CATEGORIES].index(name)
    general, vulnerable = PERSONAL[name]
    lo, hi = naqi.CATEGORIES[idx][1], naqi.CATEGORIES[idx][2]
    stages = [s for s in naqi.GRAP_STAGES if s[1] <= hi and s[2] >= lo]
    return {
        "category": name,
        "aqi_range": [lo, hi],
        "general_public": general,
        "vulnerable_groups": vulnerable,
        "you": vulnerable if profile != "general" else general,
        "you_are_vulnerable": profile != "general",
        "home_steps": [text for start, text in HOME_STEPS if idx >= start],
        "grap": [{"stage": s[0], "aqi_range": [s[1], min(s[2], 500)], "label": s[3], "measures": GOVERNMENT[s[0]]}
                 for s in stages],
        "vulnerable_definition": VULNERABLE_GROUPS,
        "sources": [ADVISORY, GRAP_SOURCE] if stages else [ADVISORY],
    }


def for_aqi(aqi: float, profile: str = "general") -> dict:
    """Measures for a specific AQI value; GRAP is narrowed to the stage that value falls in."""
    cat = naqi.category_for_aqi(aqi)["name"]
    out = for_category(cat, profile)
    stage = naqi.grap_stage(aqi)
    out["grap"] = [g for g in out["grap"] if stage and g["stage"] == stage["stage"]]
    out["aqi"] = round(aqi)
    return out


def table() -> list[dict]:
    """All six categories, for the action guide page and the API."""
    return [for_category(c[0]) for c in naqi.CATEGORIES]
