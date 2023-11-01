from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("about-us/", views.about_us, name="about_us"),
    path("how-to-use/", views.how_to_use, name="how_to_use"),
    path("konsep/", views.konsep, name="konsep"),
]
