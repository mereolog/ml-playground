from django.urls import path
from .views import load_file

urlpatterns = [
    path('load_file/', load_file, name='load_file'),
]