from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Prediction",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True, db_index=True)),
                ("source", models.CharField(choices=[("web", "Website"), ("api", "API"), ("live", "Live readings")], default="web", max_length=10)),
                ("co", models.FloatField(verbose_name="CO")),
                ("no", models.FloatField(verbose_name="NO")),
                ("no2", models.FloatField(verbose_name="NO₂")),
                ("o3", models.FloatField(verbose_name="O₃")),
                ("so2", models.FloatField(verbose_name="SO₂")),
                ("pm10", models.FloatField(verbose_name="PM10")),
                ("nh3", models.FloatField(verbose_name="NH₃")),
                ("predicted_pm25", models.FloatField(verbose_name="predicted PM2.5")),
                ("aqi", models.PositiveSmallIntegerField(verbose_name="AQI")),
                ("category", models.CharField(max_length=20)),
                ("prominent", models.CharField(max_length=10, verbose_name="prominent pollutant")),
                ("model_name", models.CharField(max_length=40)),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
    ]
