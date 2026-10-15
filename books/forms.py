from django import forms
from .models import Book

class RangeInput(forms.NumberInput):
    input_type = "range"

class BookCreateForm(forms.ModelForm):
    class Meta:
        model = Book
        fields = ["title"]
        widgets = {"title": forms.TextInput(attrs={"class": "input", "placeholder": "Ex. Histoire de Madagascar"})}

class ProcessingSettingsForm(forms.ModelForm):
    class Meta:
        model = Book
        fields = ["language", "page_format", "output_mode", "brightness", "contrast", "clahe", "background_clean", "show_through", "denoise_mode", "unsharp",
                  "auto_crop_page", "crop_margins", "crop_padding", "auto_rotate", "deskew",
                  "photo_output_mode", "photo_brightness", "photo_contrast", "photo_clahe",
                  "photo_background_clean", "photo_show_through", "photo_denoise_mode", "photo_unsharp"]
        labels = {
            "photo_output_mode": "Mode de rendu",
            "photo_brightness": "Luminosité",
            "photo_contrast": "Contraste",
            "photo_clahe": "Contraste local (CLAHE modéré)",
            "photo_background_clean": "Nettoyer le fond",
            "photo_show_through": "Atténuer le verso",
            "photo_denoise_mode": "Débruitage",
            "photo_unsharp": "Netteté (masque flou, sans halos)",
            "language": "Langue de l’OCR",
            "brightness": "Luminosité",
            "contrast": "Contraste",
            "auto_crop_page": "Recadrage automatique de la page (détecte la feuille sur la photo)",
            "crop_margins": "Recadrer les marges",
            "crop_padding": "Marge conservée autour du texte (px)",
            "auto_rotate": "Rotation automatique (pages couchées ou à l’envers)",
            "deskew": "Redresser les pages penchées",
            "output_mode": "Mode de rendu",
            "page_format": "Format de sortie du PDF",
            "show_through": "Atténuer le verso (texte vu par transparence)",
            "clahe": "Contraste local (CLAHE modéré)",
            "unsharp": "Netteté (masque flou, sans halos)",
            "background_clean": "Nettoyer le fond (ombres, papier jauni)",
            "denoise_mode": "Débruitage",
        }
        widgets = {
            "language": forms.Select(choices=[
                ("fra", "Français"),
                ("fra+eng", "Français + anglais"),
                ("eng", "Anglais"),
                ("deu", "Allemand"),
                ("deu+eng", "Allemand + anglais"),
                ("fra+deu", "Français + allemand"),
                ("deu_latf", "Allemand gothique (Fraktur)"),
                ("lat", "Latin"),
                ("fra+lat", "Français + latin"),
                ("deu+lat", "Allemand + latin"),
                ("spa", "Espagnol"),
                ("por", "Portugais"),
            ]),
            "brightness": RangeInput(attrs={"min": "0.5", "max": "1.8", "step": "0.05"}),
            "contrast": RangeInput(attrs={"min": "0.5", "max": "2.0", "step": "0.05"}),
            "show_through": RangeInput(attrs={"min": "0", "max": "1", "step": "0.05"}),
            "unsharp": RangeInput(attrs={"min": "0", "max": "2.0", "step": "0.1"}),
            "photo_brightness": RangeInput(attrs={"min": "0.5", "max": "1.8", "step": "0.05"}),
            "photo_contrast": RangeInput(attrs={"min": "0.5", "max": "2.0", "step": "0.05"}),
            "photo_show_through": RangeInput(attrs={"min": "0", "max": "1", "step": "0.05"}),
            "photo_unsharp": RangeInput(attrs={"min": "0", "max": "2.0", "step": "0.1"}),
            "crop_padding":forms.NumberInput(attrs={"min": "0", "max": "100"}),
        }
