from django.conf import settings
from django.db import models

class PriceSettings(models.Model):
    price_text = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    price_table = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    price_chart = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    price_photo = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    currency = models.CharField(max_length=8, default="€")

    @classmethod
    def load(cls):
        price_settings, _ = cls.objects.get_or_create(pk=1)
        return price_settings


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
        ("combined", "Lissage + réduction du bruit"),
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
    price_text = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    price_table = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    price_chart = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    price_photo = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    currency = models.CharField(max_length=8, default="€")
    # Manual trim of each side of the upright page, in percent: strips of the neighbouring page, shadows...
    edge_trim_left = models.FloatField(default=0.0)
    edge_trim_top = models.FloatField(default=0.0)
    edge_trim_right = models.FloatField(default=0.0)
    edge_trim_bottom = models.FloatField(default=0.0)
    a4_margin_mm =models.PositiveSmallIntegerField(default=5)   # white border around the page on the A4 sheet
    a4_align = models.CharField(max_length=10, choices=[("center", "Centrée"), ("top", "En haut")], default="center")
    lift_colored_pages = models.BooleanField(default=True)  # brighten full-colour pages (yellow, green...) photographed too dark
    neutralize_yellow_background = models.BooleanField(default=False)
    toc_links = models.BooleanField(default=True)  # make table-of-contents entries clickable in the PDF
    auto_crop_page = models.BooleanField(default=True)
    crop_margins =models.BooleanField(default=True)
    crop_padding = models.PositiveIntegerField(default=20)
    auto_rotate = models.BooleanField(default=True)
    deskew = models.BooleanField(default=True)
    output_pdf = models.FileField(upload_to="books/%Y/%m/final/", blank=True)
    error_message = models.TextField(blank=True)
    # Background job (see jobs.py). Written only with queryset.update(): a form save must never overwrite them.
    job_kind = models.CharField(max_length=10, blank=True, default="")      # "process", "analyze" or "" (none)
    job_progress = models.PositiveSmallIntegerField(default=0)              # 0-1000, a thousandth of the job
    job_step = models.CharField(max_length=200, blank=True, default="")
    job_heartbeat = models.DateTimeField(null=True, blank=True)             # last sign of life of the worker
    job_cancel = models.BooleanField(default=False)
    job_result = models.TextField(blank=True, default="")                   # "success|text" lines, shown once

    class Meta:
        ordering = ["-created_at"]

    def yield_summary(self) -> dict:
        """Pages counted by type, priced, and summed: the book's yield ("rendement").

        Pages not analysed yet have no type and no price: they are reported apart so the total is
        never mistaken for a complete one.
        """
        from collections import Counter
        from decimal import Decimal
        from .classify import LABELS, TYPES
        pages = list(self.pages.all())      # uses the prefetched pages when there are some
        counts = Counter(page.page_type for page in pages)
        rows = []
        for kind in TYPES:
            price = getattr(self, f"price_{kind}") or Decimal(0)
            rows.append({"type": kind, "label": LABELS[kind], "count": counts[kind], "price": price,
                         "subtotal": price * counts[kind]})
        total = sum((row["subtotal"] for row in rows), Decimal(0))
        analysed = sum(row["count"] for row in rows)
        return {"rows": rows, "total": total, "pages": len(pages), "analysed": analysed,
                "unanalysed": len(pages) - analysed,
                "average": (total / analysed) if analysed else Decimal(0)}

    def __str__(self):
        return self.title

class PageImage(models.Model):
    book = models.ForeignKey(Book, on_delete=models.CASCADE, related_name="pages")
    image = models.ImageField(upload_to="books/%Y/%m/pages/")
    original_name = models.CharField(max_length=255)
    order = models.PositiveIntegerField(default=0)
    rotation = models.PositiveSmallIntegerField(default=0)  # manual clockwise turn: 0, 90, 180 or 270
    has_photo = models.BooleanField(default=False)  # page with a photo: uses the book's photo_* settings
    # Trim of this page's own edges, in percent. Used instead of the book's trim when trim_custom is set.
    trim_custom = models.BooleanField(default=False)
    trim_left = models.FloatField(default=0.0)
    trim_top = models.FloatField(default=0.0)
    trim_right = models.FloatField(default=0.0)
    trim_bottom = models.FloatField(default=0.0)
    # Detected by the page analysis: "text", "chart" (graph or curve), "photo" (image) or "" (not analysed yet).
    page_type = models.CharField(max_length=10, blank=True, default="")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["order", "id"]
        unique_together = [("book", "order")]

    @property
    def custom_trim(self):
        """(left, top, right, bottom) in percent when this page has its own trim, else None."""
        if not self.trim_custom:
            return None
        return (self.trim_left, self.trim_top, self.trim_right, self.trim_bottom)

    @property
    def type_icon(self) -> str:
        from .classify import ICONS
        return ICONS.get(self.page_type, "")

    def __str__(self):
        return f"{self.book.title} — page {self.order}"
