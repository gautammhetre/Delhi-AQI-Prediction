from django.urls import path
from django.views.generic import RedirectView

from . import views

urlpatterns = [
    path("", views.now, name="now"),
    path("report/", views.report_partial, name="report"),
    path("check/", views.check, name="check"),
    path("what-to-do/", views.guide, name="guide"),
    path("ask/", views.ask, name="ask"),
    path("trends/", views.trends, name="trends"),
    path("about/", views.about, name="about"),
    path("about/model/", views.model_page, name="model"),
    path("staff/history/", views.history, name="history"),

    # old addresses keep working
    path("dashboard/", RedirectView.as_view(pattern_name="trends", permanent=True)),
    path("measures/", RedirectView.as_view(pattern_name="guide", permanent=True, query_string=True)),
    path("assistant/", RedirectView.as_view(pattern_name="ask", permanent=True, query_string=True)),
    path("model/", RedirectView.as_view(pattern_name="model", permanent=True)),
    path("history/", RedirectView.as_view(pattern_name="history", permanent=True)),
]
