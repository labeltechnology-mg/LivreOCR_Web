from django.conf import settings
from django.db import models

class Book(models.Model):
    STATUS_CHOICES = [
        ("new", "À préparer"),
        ("processing", "Traitement en cours"),
        ("done", "Terminé"),
        ("error", "Erreur"),
    ]
    OUTPUT_MODES = [
        ("color", "Couleur"),
        ("gray", "Niveaux de gris"),
        ("bw", "Noir et blanc (texte)"),
    ]
    DENOISE_MODES = [
        ("off", "Aucun"),
        ("median", "Médian léger"),
        ("edge", "Préservant les contours"),
    ]
    PAGE_FORMATS = [
        ("a4", "A4 (210 × 297 mm)"),
        ("original", "Taille d’origine de l’image"),
    ]
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="books")
    title = models.CharField("Titre du livre", max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="new")
    language = models.CharField(max_length=30, default="fra+eng")
    brightness = models.FloatField(default=1.0)
    contrast = models.FloatField(default=1.0)
    page_format = models.CharField(max_length=10, choices=PAGE_FORMATS, default="a4")
    clahe = models.BooleanField(default=False)
    unsharp = models.FloatField(default=0.0)
    show_through = models.FloatField(default=0.0)
    background_clean = models.BooleanField(default=False)
    denoise_mode = models.CharField(max_length=10, choices=DENOISE_MODES, default="off")
    output_mode = models.CharField(max_length=10, choices=OUTPUT_MODES, default="color")
    # Rendering used instead of the global settings above on pages the user marked "with photo".
    photo_output_mode = models.CharField(max_length=10, choices=OUTPUT_MODES, default="color")
    photo_brightness = models.FloatField(default=1.3)
    photo_contrast = models.FloatField(default=1.1)
    photo_clahe = models.BooleanField(default=False)
    photo_background_clean = models.BooleanField(default=False)
    photo_show_through = models.FloatField(default=0.0)
    photo_denoise_mode = models.CharField(max_length=10, choices=DENOISE_MODES, default="off")
    photo_unsharp = models.FloatField(default=0.3)
    a4_margin_mm = models.PositiveSmallIntegerField(default=5)   # white border around the page on the A4 sheet
    a4_align = models.CharField(max_length=10, choices=[("center", "Centrée"), ("top", "En haut")], default="center")
    toc_links = models.BooleanField(default=True)  # make table-of-contents entries clickable in the PDF
    auto_crop_page = models.BooleanField(default=True)
    crop_margins =models.BooleanField(default=True)
    crop_padding = models.PositiveIntegerField(default=20)
    auto_rotate = models.BooleanField(default=True)
    deskew = models.BooleanField(default=True)
    output_pdf = models.FileField(upload_to="books/%Y/%m/final/", blank=True)
    error_message = models.TextField(blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.title

class PageImage(models.Model):
    book = models.ForeignKey(Book, on_delete=models.CASCADE, related_name="pages")
    image = models.ImageField(upload_to="books/%Y/%m/pages/")
    original_name = models.CharField(max_length=255)
    order = models.PositiveIntegerField(default=0)
    rotation = models.PositiveSmallIntegerField(default=0)  # manual clockwise turn: 0, 90, 180 or 270
    has_photo = models.BooleanField(default=False)  # page with a photo: uses the book's photo_* settings
    # Detected by the page analysis: "text", "chart" (graph or curve), "photo" (image) or "" (not analysed yet).
    page_type = models.CharField(max_length=10, blank=True, default="")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["order", "id"]
        unique_together = [("book", "order")]

    @property
    def type_icon(self) -> str:
        from .classify import ICONS
        return ICONS.get(self.page_type, "")

    def __str__(self):
        return f"{self.book.title} — page {self.order}"
