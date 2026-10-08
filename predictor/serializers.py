from rest_framework import serializers

from core.assistant import PROFILES

from .models import Prediction


class PredictInputSerializer(serializers.Serializer):
    co = serializers.FloatField(min_value=0, max_value=50000, help_text="Carbon monoxide, µg/m³")
    no = serializers.FloatField(min_value=0, max_value=50000, help_text="Nitric oxide, µg/m³")
    no2 = serializers.FloatField(min_value=0, max_value=50000, help_text="Nitrogen dioxide, µg/m³")
    o3 = serializers.FloatField(min_value=0, max_value=50000, help_text="Ozone, µg/m³")
    so2 = serializers.FloatField(min_value=0, max_value=50000, help_text="Sulphur dioxide, µg/m³")
    pm10 = serializers.FloatField(min_value=0, max_value=50000, help_text="PM10, µg/m³")
    nh3 = serializers.FloatField(min_value=0, max_value=50000, help_text="Ammonia, µg/m³")
    profile = serializers.ChoiceField(choices=list(PROFILES), default="general", required=False,
                                      help_text="Who the explanation is written for")
    explain = serializers.BooleanField(default=False, required=False,
                                       help_text="Also return a plain-language explanation with sources")


class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = ["id", "created_at", "source", "co", "no", "no2", "o3", "so2", "pm10", "nh3",
                  "predicted_pm25", "aqi", "category", "prominent", "model_name"]


class AskSerializer(serializers.Serializer):
    question = serializers.CharField(max_length=500)


class LocationSerializer(serializers.Serializer):
    lat = serializers.FloatField(min_value=-90, max_value=90)
    lon = serializers.FloatField(min_value=-180, max_value=180)
    label = serializers.CharField(max_length=80, required=False, allow_blank=True)
