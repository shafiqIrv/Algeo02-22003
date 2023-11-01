from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader


# Create your views here.
def index(request):
    template = loader.get_template("pages/index.html")
    return HttpResponse(template.render())


def about_us(request):
    template = loader.get_template("pages/about_us.html")
    return HttpResponse(template.render())


def how_to_use(request):
    template = loader.get_template("pages/how_to_use.html")
    return HttpResponse(template.render())


def concept(request):
    template = loader.get_template("pages/concept.html")
    return HttpResponse(template.render())
