import numpy as np
from PIL import Image
import time
import math
import matplotlib.pyplot as plt
import os


def img_to_grayscale(image_path):
    img = np.array(Image.open(image_path))
    rgb_channels = img[..., :3] 
    
    grayscale = np.dot(rgb_channels, [0.299, 0.587, 0.114]).astype(int)
    
    return grayscale

def co_occurrence(grey_pict):
    co_occurrence = np.zeros((256, 256), dtype=int)
    for i in range (len(grey_pict)):
        for j in range (len(grey_pict[0])-1):
            co_occurrence[grey_pict[i][j]][grey_pict[i][j+1]] +=1
    co_occurrence = (co_occurrence + co_occurrence.T) / np.sum(co_occurrence)    
    return co_occurrence

def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    similarity = dot_product / (norm_a * norm_b) 
    return similarity * 100

def component(matrix):
    contrast = 0
    homogeneity = 0
    entropy = 0
    for i in range(len(matrix)):
        for j in range (len(matrix[0])):
            contrast += matrix[i][j] * ((i-j)**2)
            homogeneity += matrix[i][j] / (1 + (i-j)**2)
            if matrix[i][j] != 0:
                entropy += matrix[i][j] * math.log(matrix[i][j], 10)
    entropy = -entropy
    vector = [contrast, homogeneity, entropy]
    return vector


start = time.time()
list_files = os.listdir("big_dataset")
vectorA = component(co_occurrence(img_to_grayscale(f"ucup.png")))
sum =0
for f in list_files:
    vectorB = component(co_occurrence(img_to_grayscale(f"big_dataset/{f}")))
    pers = cosine_similarity(vectorA, vectorB)
    print("%: ", pers)
    sum += pers
end = time.time()   
print("Total Waktu:" , end -start)
print("Avg Persentage= ", sum / len(list_files))


# TESTING I : 1011.9139294624329 detik, Tanpa cosine hanya sampai component tiap matrix
# TESTING II : 948.7930908203125 detik, komparasi dengan gambar ucup, Avg Persentage =  99.93877584170228