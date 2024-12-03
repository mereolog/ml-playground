from django.urls import path
from .views import send_data, load_file

urlpatterns = [
    path('load_file/', load_file, name='load_file'),
    path('send_data/', send_data, name='send_data'),

    
]
