from django.contrib import admin

from .models import Prediction


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ("created_at", "source", "aqi", "category", "prominent", "predicted_pm25", "pm10", "model_name")
    list_filter = ("category", "source", "model_name")
    date_hierarchy = "created_at"
    readonly_fields = [f.name for f in Prediction._meta.fields]
    search_fields = ("category", "prominent")
