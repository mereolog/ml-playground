from django.urls import path

from .views import algorithm_form

urlpatterns = [
    path("algorithm-form/<str:algorithm_name>/", algorithm_form, name="algorithm_form"),]
    