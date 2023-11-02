from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader


# Create your views here.
def index(request):
    return render(request, "pages/index.html")


def about_us(request):
    return render(request, "pages/about_us.html")


def how_to_use(request):
    return render(request, "pages/how_to_use.html")


def concept(request):
    return render(request, "pages/concept.html")
