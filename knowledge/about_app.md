---
title: How this app's predictions work and their limits
source: Delhi AQI project documentation
url: /about/model/
---

## What the prediction model does
The model estimates the PM2.5 concentration for an hour in Delhi from the other pollutants measured in that hour: CO, NO, NO2, O3, SO2, PM10 and NH3. It is a gradient-boosting regression model trained on about 15,000 hours and tested on the most recent five months, which it never saw during training. The AQI category is then calculated from the predicted PM2.5 and the other pollutants using India's NAQI breakpoints.

## How accurate the prediction is
On the held-out test months the typical error is about 11 µg/m³ of PM2.5, and the predicted category matches the true category about 93% of the time. A simple rule that treats PM2.5 as a fixed share of PM10 has an error of about 16 µg/m³, so the model is about a third more accurate than that baseline.

## Limits of this app
The official AQI uses 24-hour averages (8-hour for CO and ozone), but this app works with single hourly readings, so its AQI is indicative only. It is not a forecast of future air quality. It is not medical advice: anyone with symptoms should see a doctor. Live readings come from a modelled air-quality service, and ammonia is not available there for India, so a typical Delhi value is used for it.
