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
        fields = ["language", "page_format", "a4_margin_mm", "a4_align", "toc_links",
                  "edge_trim_left", "edge_trim_top", "edge_trim_right", "edge_trim_bottom", "output_mode", "brightness", "contrast", "clahe", "background_clean", "show_through", "denoise_mode", "unsharp",
                  "auto_crop_page", "crop_margins", "crop_padding", "auto_rotate", "deskew",
                  "photo_output_mode", "photo_brightness", "photo_contrast", "photo_clahe",
                  "photo_background_clean", "photo_show_through", "photo_denoise_mode", "photo_unsharp"]
        labels = {
            "edge_trim_left": "Gauche (%)",
            "edge_trim_top": "Haut (%)",
            "edge_trim_right": "Droite (%)",
            "edge_trim_bottom": "Bas (%)",
            "a4_margin_mm": "Marge autour de la page sur l’A4 (mm)",
            "a4_align": "Position de la page sur l’A4",
            "toc_links": "Table des matières cliquable (liens et signets dans le PDF)",
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
            "edge_trim_left": RangeInput(attrs={"min": "0", "max": "25", "step": "0.5"}),
            "edge_trim_top": RangeInput(attrs={"min": "0", "max": "25", "step": "0.5"}),
            "edge_trim_right": RangeInput(attrs={"min": "0", "max": "25", "step": "0.5"}),
            "edge_trim_bottom": RangeInput(attrs={"min": "0", "max": "25", "step": "0.5"}),
            "a4_margin_mm": forms.NumberInput(attrs={"min": "0", "max": "30"}),
            "crop_padding":forms.NumberInput(attrs={"min": "0", "max": "100"}),
        }
