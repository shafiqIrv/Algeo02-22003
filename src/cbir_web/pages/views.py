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

    if len(img.shape) == 2:
        img = np.array([img])

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


# Mengubah data RGB dari suatu image menjadi data histogram H, S, dan V.
# Menerima masukan berupa image dengan format array of RGB
# Menghasilkan keluaran berupa histogram H, S, dan V
def rgb_to_hsv_hist(rgb_image):
    # Normalize RGB values to the range [0, 1]
    normalized_image = rgb_image / 255.0

    # Find Cmax, Cmin, and ∆
    Cmax = np.max(normalized_image, axis=-1)
    Cmin = np.min(normalized_image, axis=-1)
    delta = Cmax - Cmin

    # Calculate Hue (H)
    H = np.zeros_like(Cmax)
    non_zero_delta = delta != 0

    H[non_zero_delta] = np.where(
        Cmax[non_zero_delta] == normalized_image[non_zero_delta, ..., 0],
        (
            (
                (
                    normalized_image[non_zero_delta, ..., 1]
                    - normalized_image[non_zero_delta, ..., 2]
                )
                / delta[non_zero_delta]
            )
            % 6.0
        ),
        H[non_zero_delta],
    )

    H[non_zero_delta] = np.where(
        Cmax[non_zero_delta] == normalized_image[non_zero_delta, ..., 1],
        (
            2.0
            + (
                normalized_image[non_zero_delta, ..., 2]
                - normalized_image[non_zero_delta, ..., 0]
            )
            / delta[non_zero_delta]
        ),
        H[non_zero_delta],
    )

    H[non_zero_delta] = np.where(
        Cmax[non_zero_delta] == normalized_image[non_zero_delta, ..., 2],
        (
            4.0
            + (
                normalized_image[non_zero_delta, ..., 0]
                - normalized_image[non_zero_delta, ..., 1]
            )
            / delta[non_zero_delta]
        ),
        H[non_zero_delta],
    )

    H = (H / 6.0) % 1.0

    # Calculate Saturation (S)
    S = np.zeros_like(Cmax)
    S[Cmax != 0] = delta[Cmax != 0] / Cmax[Cmax != 0]

    # Calculate Value (V)
    V = Cmax

    hist_h = np.histogram(
        H,
        bins=[
            0,
            25 / 360,
            40 / 360,
            120 / 360,
            190 / 360,
            270 / 360,
            295 / 360,
            315 / 360,
            360 / 360,
        ],
    )
    hist_s = np.histogram(S, bins=[0, 0.2, 0.7, 1])
    hist_v = np.histogram(V, bins=[0, 0.2, 0.7, 1])

    return hist_h, hist_s, hist_v


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

                if request.POST["cbir_mode"] == "color":
                    start = time.time()

                    imgA = Image.open(MEDIA_ROOT + "test.jpg")
                    test_h, test_s, test_v = rgb_to_hsv_hist(np.array(imgA))

                    if "color.json" not in json_dir:
                        list_files = os.listdir(MEDIA_ROOT + "dataset/")
                        for f in list_files:
                            imgB = Image.open(MEDIA_ROOT + "dataset/" + f)
                            data_h, data_s, data_v = rgb_to_hsv_hist(np.array(imgB))

                            similarity_h = cosine_similarity(test_h[0], data_h[0])
                            similarity_s = cosine_similarity(test_s[0], data_s[0])
                            similarity_v = cosine_similarity(test_v[0], data_v[0])

                            avg_similarity = (
                                similarity_h + similarity_s + similarity_v
                            ) / 3

                            avg_similarity = float(str(avg_similarity)[:5])

                            if avg_similarity > 60:
                                query.append(
                                    {
                                        "file": "media/dataset/" + f,
                                        "hist_h": data_h[0].tolist(),
                                        "hist_s": data_s[0].tolist(),
                                        "hist_v": data_v[0].tolist(),
                                        "percentage": avg_similarity,
                                    }
                                )

                        with open(JSON_ROOT + "color.json", "w") as f:
                            json.dump(query, f)
                    else:
                        with open(JSON_ROOT + "color.json", "r") as f:
                            data = json.load(f)
                            for d in data:
                                data_h = d["hist_h"]
                                data_s = d["hist_s"]
                                data_v = d["hist_v"]

                                similarity_h = cosine_similarity(test_h[0], data_h)
                                similarity_s = cosine_similarity(test_s[0], data_s)
                                similarity_v = cosine_similarity(test_v[0], data_v)

                                avg_similarity = (
                                    similarity_h + similarity_s + similarity_v
                                ) / 3

                                avg_similarity = float(str(avg_similarity)[:5])

                                if avg_similarity > 60:
                                    query.append(
                                        {
                                            "file": d["file"],
                                            "percentage": avg_similarity,
                                        }
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
                elif request.POST["cbir_mode"] == "texture":
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

                jsons = os.listdir(JSON_ROOT)
                if "color.json" in jsons:
                    os.remove(JSON_ROOT + "color.json")
                if "texture.json" in jsons:
                    os.remove(JSON_ROOT + "texture.json")
                if "result.json" in jsons:
                    os.remove(JSON_ROOT + "result.json")
                if "time.json" in jsons:
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
