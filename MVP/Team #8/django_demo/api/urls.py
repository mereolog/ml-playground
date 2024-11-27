from django.urls import path
from .views import send_data

urlpatterns = [
    path('send_data/', send_data, name='send_data'),

    
]
