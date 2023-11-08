from django.db import models

# Create your models here.


def content_file_name(instance, filename):
    return "test.jpg"


class Image(models.Model):
    file = models.ImageField(upload_to=content_file_name)


class Dataset(models.Model):
    file = models.ImageField(upload_to="dataset/")
