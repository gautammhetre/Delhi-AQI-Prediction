from django import template

from core import naqi

register = template.Library()


@register.filter
def cat_slug(name):
    """'Very Poor' -> 'very-poor', used as a CSS class for the official NAQI colours."""
    return str(name).lower().replace(" ", "-")


@register.filter
def aqi_slug(aqi):
    return cat_slug(naqi.category_for_aqi(float(aqi))["name"])


@register.filter
def get(mapping, key):
    try:
        return mapping.get(key)
    except AttributeError:
        return None


@register.filter
def pct_of(value, total):
    try:
        return min(100.0, round(100 * float(value) / float(total), 1))
    except (TypeError, ValueError, ZeroDivisionError):
        return 0


@register.filter
def pollutant_label(key):
    return naqi.LABELS.get(key, key)


@register.filter
def percent(value, digits=0):
    try:
        return f"{float(value) * 100:.{int(digits)}f}%"
    except (TypeError, ValueError):
        return ""


@register.filter
def absval(value):
    try:
        return abs(float(value))
    except (TypeError, ValueError):
        return 0


@register.filter
def index(sequence, i):
    try:
        return sequence[int(i)]
    except (IndexError, TypeError, ValueError):
        return None


@register.filter
def category_health(name):
    """CPCB health-impact statement for a category name."""
    for cat, _lo, _hi, health in naqi.CATEGORIES:
        if cat == name:
            return health
    return ""


@register.filter
def last_hour(outlook):
    try:
        return outlook[-1]["hour"]
    except (IndexError, KeyError, TypeError):
        return ""


# Gauge geometry: a 270° arc on a circle of radius 52 (see _aqi_hero.html).
_GAUGE_LEN = 2 * 3.141592653589793 * 52 * 0.75


@register.filter
def gauge_dash(aqi):
    """stroke-dasharray for the filled part of the AQI gauge (0-500)."""
    try:
        p = max(0.0, min(float(aqi), 500.0)) / 500.0
    except (TypeError, ValueError):
        p = 0.0
    return f"{_GAUGE_LEN * p:.1f} 999"


@register.simple_tag
def gauge_track():
    return f"{_GAUGE_LEN:.1f} 999"


@register.filter
def cat_var(name):
    """'Very Poor' -> 'var(--very-poor)' for inline colour."""
    return f"var(--{cat_slug(name)})"
