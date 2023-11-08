from django import forms
from .models import Image


class ImageForm(forms.ModelForm):
    image_form = forms.BooleanField(
        widget=forms.HiddenInput, initial=True, required=False
    )

    class Meta:
        model = Image
        fields = ["file"]


class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class MultipleFileField(forms.ImageField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput())
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        single_file_clean = super().clean
        if isinstance(data, (list, tuple)):
            result = [single_file_clean(d, initial) for d in data]
        else:
            result = single_file_clean(data, initial)
        return result


class UploadFileForm(forms.Form):
    dataset_form = forms.BooleanField(
        widget=forms.HiddenInput, initial=True, required=False
    )
    files = MultipleFileField()
