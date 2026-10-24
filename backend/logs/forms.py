from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import UserCreationForm
from django.utils import timezone

from .models import BabyLog


class RegistrationForm(UserCreationForm):
    class Meta:
        model = get_user_model()
        fields = ("username",)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["username"].label = "ユーザー名"
        self.fields["password1"].label = "パスワード"
        self.fields["password2"].label = "パスワード（確認）"
        for field in self.fields.values():
            field.widget.attrs["class"] = "form-control"


class BabyLogForm(forms.ModelForm):
    class Meta:
        model = BabyLog
        fields = (
            "log_type",
            "timestamp",
            "amount",
            "duration",
            "temperature",
            "memo",
        )
        widgets = {
            "timestamp": forms.DateTimeInput(
                format="%Y-%m-%dT%H:%M",
                attrs={"type": "datetime-local", "class": "form-control"},
            ),
            "amount": forms.NumberInput(
                attrs={
                    "placeholder": "ml",
                    "min": "0",
                    "step": "1",
                    "inputmode": "decimal",
                    "class": "form-control",
                }
            ),
            "duration": forms.NumberInput(
                attrs={
                    "placeholder": "分",
                    "min": "0",
                    "step": "1",
                    "inputmode": "numeric",
                    "class": "form-control",
                }
            ),
            "temperature": forms.NumberInput(
                attrs={
                    "placeholder": "℃",
                    "step": "0.1",
                    "inputmode": "decimal",
                    "class": "form-control",
                }
            ),
            "log_type": forms.Select(attrs={"class": "form-select"}),
            "memo": forms.Textarea(attrs={"class": "form-control", "rows": 3}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.instance.pk:
            self.initial.setdefault("timestamp", timezone.localtime())
        else:
            self.initial.setdefault(
                "timestamp",
                timezone.localtime(self.instance.timestamp),
            )
        self.fields["timestamp"].input_formats = [
            "%Y-%m-%dT%H:%M",
            *self.fields["timestamp"].input_formats,
        ]

    def clean(self):
        cleaned_data = super().clean()
        log_type = cleaned_data.get("log_type")

        if log_type != "milk":
            cleaned_data["amount"] = None
        if log_type not in {"milk", "sleep"}:
            cleaned_data["duration"] = None
        if log_type != "temperature":
            cleaned_data["temperature"] = None
        return cleaned_data