from django.urls import path
from .views import update_plot

urlpatterns = [
    path('plot/<str:plot>/', update_plot, name='plotly'),
    
]
