from django.urls import path

from . import api

urlpatterns = [
    path("predict", api.PredictView.as_view(), name="api-predict"),
    path("history", api.HistoryView.as_view(), name="api-history"),
    path("stats", api.StatsView.as_view(), name="api-stats"),
    path("live", api.LiveView.as_view(), name="api-live"),
    path("report", api.ReportView.as_view(), name="api-report"),
    path("places", api.PlacesView.as_view(), name="api-places"),
    path("measures", api.MeasuresView.as_view(), name="api-measures"),
    path("ask", api.AskView.as_view(), name="api-ask"),
]
