from django.urls import path

from .views import linear_regression, algorithm_form

urlpatterns = [path("linear_regression/", linear_regression, name="linear-regression"),
                path("algorithm-form/<str:algorithm_name>/", algorithm_form, name="algorithm_form"),]
    