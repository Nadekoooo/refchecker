from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_and_predict, name='upload'),  # Halaman utama upload & prediksi
]
