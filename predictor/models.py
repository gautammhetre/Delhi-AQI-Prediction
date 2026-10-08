from django.db import models


class Prediction(models.Model):
    """Every prediction made through the website or the API, so it can be reviewed later
    (History page, admin panel, /api/v1/history)."""

    class Source(models.TextChoices):
        WEB = "web", "Website"
        API = "api", "API"
        LIVE = "live", "Live readings"

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    source = models.CharField(max_length=10, choices=Source.choices, default=Source.WEB)

    # inputs, µg/m³
    co = models.FloatField("CO")
    no = models.FloatField("NO")
    no2 = models.FloatField("NO₂")
    o3 = models.FloatField("O₃")
    so2 = models.FloatField("SO₂")
    pm10 = models.FloatField("PM10")
    nh3 = models.FloatField("NH₃")

    # outputs
    predicted_pm25 = models.FloatField("predicted PM2.5")
    aqi = models.PositiveSmallIntegerField("AQI")
    category = models.CharField(max_length=20)
    prominent = models.CharField("prominent pollutant", max_length=10)
    model_name = models.CharField(max_length=40)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"AQI {self.aqi} ({self.category}) at {self.created_at:%Y-%m-%d %H:%M}"
