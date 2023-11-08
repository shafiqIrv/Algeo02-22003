from django.shortcuts import render
from .forms import ImageForm, UploadFileForm
from .models import Dataset
import os

MEDIA_ROOT = "pages/static/media/"


# Create your views here.
def index(request):
    files = Dataset.objects.all()
    form = ImageForm(request.POST, request.FILES)
    dataset_form = UploadFileForm(request.POST, request.FILES)
    if request.method == "POST":
        if "image_form" in request.POST:
            if form.is_valid():
                test_jpg = os.listdir(MEDIA_ROOT)
                if len(test_jpg) > 1:
                    os.remove(MEDIA_ROOT + test_jpg[1])
                form.save()
                return render(
                    request,
                    "pages/index.html",
                    context={
                        "form": form,
                        "dataset_form": dataset_form,
                        "image": "media/test.jpg",
                        "files": files,
                    },
                )
        elif "dataset_form" in request.POST:
            if dataset_form.is_valid():
                datasets = os.listdir(MEDIA_ROOT + "dataset/")
                if len(datasets) > 0:
                    for image in datasets:
                        os.remove(MEDIA_ROOT + "dataset/" + image)

                for uploaded_file in request.FILES.getlist("files"):
                    Dataset.objects.create(file=uploaded_file)
                return render(
                    request,
                    "pages/index.html",
                    context={
                        "form": form,
                        "dataset_form": dataset_form,
                        "image": "media/test.jpg",
                        "files": files,
                    },
                )
    else:
        form = ImageForm()

    return render(
        request,
        "pages/index.html",
        context={
            "form": form,
            "dataset_form": dataset_form,
            "image": "media/test.jpg",
            "files": files,
        },
    )


def about_us(request):
    return render(request, "pages/about_us.html")


def how_to_use(request):
    return render(request, "pages/how_to_use.html")


def concept(request):
    return render(request, "pages/concept.html")
