from django import forms

from core.assistant import PROFILES
from core.ml import FEATURES

LABELS = {"co": "CO", "no": "NO", "no2": "NO₂", "o3": "O₃", "so2": "SO₂", "pm10": "PM10", "nh3": "NH₃"}
NAMES = {"co": "carbon monoxide", "no": "nitric oxide", "no2": "nitrogen dioxide", "o3": "ozone",
         "so2": "sulphur dioxide", "pm10": "coarse particles", "nh3": "ammonia"}
PROFILE_LABELS = {
    "general": "Healthy adult", "children": "Parent of young children", "elderly": "Older adult",
    "lung_heart": "Asthma, lung or heart condition", "pregnant": "Pregnant", "outdoor_worker": "Outdoor worker",
}


class PredictForm(forms.Form):
    """Pollutant readings in µg/m³. Django validates each value on the server
    (numeric, not negative, not absurdly large) before the model ever sees it."""

    profile = forms.ChoiceField(choices=[(k, PROFILE_LABELS[k]) for k in PROFILES], initial="general",
                                label="Advice for")

    def __init__(self, *args, stats=None, **kwargs):
        super().__init__(*args, **kwargs)
        for f in FEATURES:
            help_text = f"{NAMES[f]}, µg/m³"
            if stats:
                help_text += f" · typical {stats[f]['median']:.0f}"
            self.fields[f] = forms.FloatField(label=LABELS[f], min_value=0, max_value=50000, help_text=help_text,
                                              widget=forms.NumberInput(attrs={"step": "any", "inputmode": "decimal"}))
        # put the profile selector last
        self.order_fields(FEATURES + ["profile"])

    def values(self) -> dict:
        return {f: self.cleaned_data[f] for f in FEATURES}


class LocationForm(forms.Form):
    lat = forms.FloatField(min_value=-90, max_value=90)
    lon = forms.FloatField(min_value=-180, max_value=180)
    label = forms.CharField(max_length=80, required=False)


class AskForm(forms.Form):
    question = forms.CharField(max_length=500, label="Your question",
                               widget=forms.TextInput(attrs={"placeholder": "e.g. Is it safe for my kids to play outside at Very Poor AQI?",
                                                             "autocomplete": "off"}))
