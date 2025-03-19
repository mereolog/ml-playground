from django.urls import path

from .views import linear_regression

urlpatterns = [path("linear_regression/", linear_regression, name="linear-regression")]
