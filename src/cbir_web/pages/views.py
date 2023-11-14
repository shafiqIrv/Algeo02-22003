from django.shortcuts import render
from .forms import ImageForm, UploadFileForm
from .models import Dataset
import os
from django.core.paginator import Paginator

import numpy as np
from PIL import Image
import time
import json


MEDIA_ROOT = "pages/static/media/"
JSON_ROOT = "pages/static/json/"


# Texture
def img_to_grayscale(image_path):
    img = np.array(Image.open(image_path))
    rgb_channels = img[..., :3]

    grayscale = np.dot(rgb_channels, [0.299, 0.587, 0.114]).astype(int)

    return grayscale


def co_occurrence(grey_pict):
    co_occurrence = np.zeros((256, 256), dtype=int)

    grey_pict = np.array(grey_pict)
    height, width = grey_pict.shape

    for i in range(height):
        for j in range(width - 1):
            co_occurrence[grey_pict[i][j]][grey_pict[i][j + 1]] += 1
    co_occurrence = (co_occurrence + co_occurrence.T) / np.sum(co_occurrence)
    return co_occurrence


def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    similarity = dot_product / (norm_a * norm_b)
    return similarity * 100


def component(matrix):
    matrix = np.array(matrix)

    i, j = np.indices(matrix.shape)
    diff = i - j

    # asm = np.sum(pow(matrix,2))
    contrast = np.sum(matrix * pow(diff, 2))
    homogeneity = np.sum(matrix / (1 + pow(diff, 2)))

    nonzero_elements = matrix[matrix != 0]
    entropy = -np.sum(nonzero_elements * np.log10(nonzero_elements))

    return [contrast, homogeneity, entropy]


# Create your views here.
def index(request):
    files = Dataset.objects.all()
    form = ImageForm(request.POST, request.FILES)
    dataset_form = UploadFileForm(request.POST, request.FILES)

    list_dir = os.listdir(JSON_ROOT)

    query = list()

    if "result.json" in list_dir:
        with open(JSON_ROOT + "result.json", "r") as f:
            result = json.load(f)
            paginator = Paginator(result, 6)
            page_number = request.GET.get("page")
            page_obj = paginator.get_page(page_number)

    if "time.json" in list_dir:
        with open(JSON_ROOT + "time.json", "r") as f:
            time_taken = json.load(f)

    if request.method == "POST":
        if "image_form" in request.POST:
            if form.is_valid():
                test_jpg = os.listdir(MEDIA_ROOT)
                if len(test_jpg) > 1:
                    os.remove(MEDIA_ROOT + test_jpg[1])
                form.save()

                json_dir = os.listdir(JSON_ROOT)

                if request.POST["cbir_mode"] == "texture":
                    start = time.time()

                    vectorA = component(
                        co_occurrence(img_to_grayscale(MEDIA_ROOT + "test.jpg"))
                    )

                    if "texture.json" not in json_dir:
                        list_files = os.listdir(MEDIA_ROOT + "dataset/")
                        for f in list_files:
                            vectorB = component(
                                co_occurrence(
                                    img_to_grayscale(MEDIA_ROOT + "dataset/" + f)
                                )
                            )
                            pers = float(str(cosine_similarity(vectorA, vectorB))[:5])

                            if pers > 60:
                                query.append(
                                    {
                                        "file": "media/dataset/" + f,
                                        "hist": vectorB,
                                        "percentage": pers,
                                    }
                                )

                        with open(JSON_ROOT + "texture.json", "w") as f:
                            json.dump(query, f)
                    else:
                        with open(JSON_ROOT + "texture.json", "r") as f:
                            data = json.load(f)
                            for d in data:
                                vectorB = d["hist"]
                                pers = float(
                                    str(cosine_similarity(vectorA, vectorB))[:5]
                                )

                                if pers > 60:
                                    query.append(
                                        {"file": d["file"], "percentage": pers}
                                    )

                    result = list(
                        sorted(query, key=lambda i: i["percentage"], reverse=True)
                    )

                    with open(JSON_ROOT + "result.json", "w") as f:
                        json.dump(result, f)

                    paginator = Paginator(result, 6)
                    page_number = request.GET.get("page")
                    page_obj = paginator.get_page(page_number)

                    end = time.time()

                    time_taken = round((end - start), 2)

                    with open(JSON_ROOT + "time.json", "w") as f:
                        json.dump(time_taken, f)

                    return render(
                        request,
                        "pages/index.html",
                        context={
                            "form": form,
                            "dataset_form": dataset_form,
                            "image": "media/test.jpg",
                            "page_obj": page_obj,
                            "time_taken": time_taken,
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

                os.remove(JSON_ROOT + "texture.json")
                os.remove(JSON_ROOT + "result.json")
                os.remove(JSON_ROOT + "time.json")

                return render(
                    request,
                    "pages/index.html",
                    context={
                        "form": form,
                        "dataset_form": dataset_form,
                        "image": "media/test.jpg",
                    },
                )
    else:
        form = ImageForm()
        dataset_form = UploadFileForm()

    return render(
        request,
        "pages/index.html",
        context={
            "form": form,
            "dataset_form": dataset_form,
            "image": "media/test.jpg",
            "page_obj": page_obj,
            "time_taken": time_taken,
        },
    )


def about_us(request):
    return render(request, "pages/about_us.html")


def how_to_use(request):
    return render(request, "pages/how_to_use.html")


def concept(request):
    return render(request, "pages/concept.html")
