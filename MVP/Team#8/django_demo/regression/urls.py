from django.urls import path
from .views import index, compute_regression, regression, redis_post

urlpatterns = [
    path('', index, name='index'),
    path('redis-post/<str:app>/<str:key>/', redis_post, name='redis_post'),
    path('regression/', regression, name='regression'),
    path('compute/', compute_regression, name='compute_regression'),
    
]
