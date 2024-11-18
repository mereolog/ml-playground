from django.urls import path
from .views import index, compute_regression, regression

urlpatterns = [
    path('', index, name='index'),
    path('regression/', regression, name='regression'),
    path('compute/', compute_regression, name='compute_regression'),
]
