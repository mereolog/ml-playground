from django.urls import path
from .views import index, compute_regression

urlpatterns = [
    path('', index, name='index'),
    path('compute/', compute_regression, name='compute_regression'),
]
