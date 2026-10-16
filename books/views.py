import base64
import io
import json
import logging
from pathlib import Path
import tempfile
from django.contrib import messages
from django.core.files.storage import default_storage
from django.contrib.auth.decorators import login_required
from django.db.models import Count, Q
from django.http import FileResponse, Http404, HttpResponse, JsonResponse
from PIL import Image, ImageOps
from .classify import CHART, PHOTO, TABLE, TEXT, TYPES, classify_page
from .image_processing import (
    apply_page_layout, crop_to_page, prepare_page_image, process_image, same_side_neighbours,
    sibling_quad,
)
from django.utils.text import get_valid_filename
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST
from . import jobs
from .forms import BookCreateForm, PriceSettingsForm, ProcessingSettingsForm
from .llm import OllamaRecommendationError, recommend_page_settings
from .models import Book, PageImage, PriceSettings
from .ocr import build_searchable_pdf, OCRProcessingError
from .text_layers import TextLayerError, separate_text_background

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
MAX_FILES_PER_UPLOAD = 500
MAX_FILE_SIZE = 30 * 1024 * 1024
logger = logging.getLogger(__name__)

@login_required
def media_file(request, path):
    """Serve an uploaded page or a generated PDF, only to the book's owner."""
    owned = (PageImage.objects.filter(image=path, book__owner=request.user).exists()
             or Book.objects.filter(output_pdf=path, owner=request.user).exists())
    if not owned:
        raise Http404("Fichier introuvable.")
    try:
        return FileResponse(default_storage.open(path, "rb"))
    except (FileNotFoundError, ValueError):
        raise Http404("Fichier introuvable.")

@login_required
def dashboard(request):
    prices = PriceSettings.load()
    if request.method == "POST":
        price_form = PriceSettingsForm(request.POST, instance=prices)
        if price_form.is_valid():
            price_form.save()
            messages.success(request, "Tarifs globaux enregistrés.")
            return redirect("dashboard")
    else:
        price_form = PriceSettingsForm(instance=prices)
    books = list(
        Book.objects.filter(owner=request.user)
        .annotate(page_count=Count("pages"))
        .prefetch_related("pages")
    )
    for book in books:
        book.yield_info = book.yield_summary()
    summary = Book.objects.filter(owner=request.user).aggregate(
        book_count=Count("pk", distinct=True),
        page_count=Count("pages"),
        ready_count=Count("pk", filter=Q(status="done"), distinct=True),
    )
    return render(request, "books/dashboard.html", {
        "books": books,
        "summary": summary,
        "price_form": price_form,
        "prices": prices,
    })

@login_required
def create_book(request):
    if request.method == "POST":
        form = BookCreateForm(request.POST)
        if form.is_valid():
            book = form.save(commit=False)
            book.owner = request.user
            prices = PriceSettings.load()
            for field in ("price_text", "price_table", "price_chart", "price_photo", "currency"):
                setattr(book, field, getattr(prices, field))
            book.save()
            messages.success(request, "Livre créé. Vous pouvez maintenant importer les pages.")
            return redirect("book_detail", pk=book.pk)
    else:
        form = BookCreateForm()
    return render(request, "books/create_book.html", {"form": form})

@login_required
def book_detail(request, pk):
    book = jobs.recover_if_stale(get_object_or_404(Book, pk=pk, owner=request.user))
    # What the last background job left to say (done, cancelled, failed) is shown once.
    for level, text in jobs.take_result(book):
        getattr(messages, level if level in ("success", "warning", "error") else "info")(request, text)
    settings_form = ProcessingSettingsForm(instance=book)
    pages = list(book.pages.all())
    preview_pages = {str(p.pk): p.image.url for p in pages}
    return render(request, "books/book_detail.html", {
        "book": book, "settings_form": settings_form, "pages": pages,
        "preview_pages_json": json.dumps(preview_pages).replace("<", "\\u003c"),
        "photo_count": sum(1 for p in pages if p.has_photo),
        "edge_trimmed": any((book.edge_trim_left, book.edge_trim_top, book.edge_trim_right, book.edge_trim_bottom)),
        "type_counts": {
            "text": sum(1 for p in pages if p.page_type == TEXT),
            "table": sum(1 for p in pages if p.page_type == TABLE),
            "chart": sum(1 for p in pages if p.page_type == CHART),
            "photo": sum(1 for p in pages if p.page_type == PHOTO),
            "unknown": sum(1 for p in pages if not p.page_type),
        },
        "yield": book.yield_summary(),
        "job": jobs.state(book),
    })

def _refuse_while_busy(request, book, what):
    """Messages and a redirect when a background job is using the book, else None."""
    if book.job_kind:
        messages.error(request, f"Un traitement est en cours sur ce livre : {what} après sa fin ou son annulation.")
        return redirect("book_detail", pk=book.pk)
    return None


def _refuse_json_while_busy(book):
    if book.job_kind:
        return JsonResponse({"error": "Un traitement est en cours sur ce livre."}, status=409)
    return None


@login_required
@require_POST
def upload_pages(request, pk):
    book = jobs.recover_if_stale(get_object_or_404(Book, pk=pk, owner=request.user))
    busy = _refuse_while_busy(request, book, "importez les pages")
    if busy:
        return busy
    uploads = request.FILES.getlist("images")
    if not uploads:
        messages.error(request, "Sélectionnez au moins une image.")
        return redirect("book_detail", pk=pk)
    if len(uploads) > MAX_FILES_PER_UPLOAD:
        messages.error(request, f"Maximum {MAX_FILES_PER_UPLOAD} images par import.")
        return redirect("book_detail", pk=pk)

    current_max = book.pages.order_by("-order").values_list("order", flat=True).first() or 0
    created = 0
    for upload in uploads:
        ext = Path(upload.name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            messages.warning(request, f"Format refusé : {upload.name}")
            continue
        if upload.size > MAX_FILE_SIZE:
            messages.warning(request, f"Fichier trop volumineux (30 Mo max) : {upload.name}")
            continue
        # Pillow verifies the actual image format rather than trusting extension only.
        try:
            from PIL import Image
            upload.seek(0)
            with Image.open(upload) as image:
                image.verify()
            upload.seek(0)
        except Exception:
            messages.warning(request, f"Image invalide : {upload.name}")
            continue
        current_max += 1
        PageImage.objects.create(book=book, image=upload, original_name=upload.name, order=current_max)
        created += 1

    if created:
        book.status = "new"
        book.save(update_fields=["status", "updated_at"])
        messages.success(request, f"{created} image(s) importée(s).")
    return redirect("book_detail", pk=pk)

@login_required
def preview_page(request, pk, page_pk):
    """Render one page with the settings from the query string, without saving anything."""
    book = get_object_or_404(Book, pk=pk, owner=request.user)
    page = get_object_or_404(PageImage, pk=page_pk, book=book)
    # Unsaved copy so the stored settings stay untouched until the user saves.
    draft = Book(owner=book.owner, title=book.title)
    # A page opened before an update may not send the newest settings: use the saved value for
    # any setting that is missing, instead of refusing the preview. (An unchecked box is simply
    # absent from a form, so checkboxes are left as they are.)
    data = request.GET.copy()
    for name, field in ProcessingSettingsForm().fields.items():
        if name not in data and field.widget.input_type != "checkbox":
            data[name] = getattr(book, name)
    form = ProcessingSettingsForm(data, instance=draft)
    if not form.is_valid():
        return HttpResponse("Paramètres invalides.", status=400)
    form.save(commit=False)
    try:
        # Larger previews are requested when the user zooms in.
        try:
            size = min(max(int(request.GET.get("size", 1000)), 600), 2400)
        except ValueError:
            size = 1000
        neighbours = same_side_neighbours(list(book.pages.all()), page) if draft.auto_crop_page else None
        adjusted, base_side = prepare_page_image(
            Path(page.image.path), draft, max_side=size, extra_rotation=page.rotation,
            fallback_sources=neighbours, photo_page=page.has_photo, page_trim=page.custom_trim,
        )
        rendered = apply_page_layout(adjusted.copy(), draft, max_side=size, base_side=base_side)
        previews = {}
        for name, image in (("adjusted", adjusted), ("rendered", rendered)):
            buffer = io.BytesIO()
            image.save(buffer, "JPEG", quality=85)
            previews[name] = f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"
        if request.GET.get("layers") == "1":
            text, background, overlay = separate_text_background(adjusted)
            for name, layer, image_format in (
                ("text", text, "PNG"),
                ("background", background, "JPEG"),
                ("overlay", overlay, "JPEG"),
            ):
                buffer = io.BytesIO()
                layer.save(buffer, image_format, **({"quality": 85} if image_format == "JPEG" else {}))
                mime_type = "image/png" if image_format == "PNG" else "image/jpeg"
                previews[name] = f"data:{mime_type};base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"
    except TextLayerError as exc:
        return JsonResponse({"error": str(exc)}, status=exc.status_code)
    except Exception:
        logger.exception("Preview generation failed for book %s page %s", pk, page_pk)
        return HttpResponse("Impossible de générer l’aperçu.", status=500)
    response = JsonResponse(previews)
    response["Cache-Control"] = "no-store"
    return response

@login_required
@require_POST
def recommend_settings(request, pk, page_pk):
    book = get_object_or_404(Book, pk=pk, owner=request.user)
    page = get_object_or_404(PageImage, pk=page_pk, book=book)
    try:
        recommendation = recommend_page_settings(page)
    except OllamaRecommendationError as exc:
        return JsonResponse({"error": str(exc)}, status=502)
    return JsonResponse(recommendation)

@login_required
@require_POST
def rotate_page(request, pk, page_pk):
    """Turn one page by 90 degrees, on top of the automatic rotation."""
    book = jobs.recover_if_stale(get_object_or_404(Book, pk=pk, owner=request.user))
    busy = _refuse_json_while_busy(book)
    if busy:
        return busy
    page = get_object_or_404(PageImage, pk=page_pk, book=book)
    step = -90 if request.POST.get("direction") == "ccw" else 90
    page.rotation = (page.rotation + step) % 360
    page.save(update_fields=["rotation"])
    return JsonResponse({"rotation": page.rotation})

@login_required
@require_POST
def analyze_pages(request, pk):
    """Start the analysis of every page (text, table, chart, image) in the background."""
    book = jobs.recover_if_stale(get_object_or_404(Book, pk=pk, owner=request.user))
    busy = _refuse_while_busy(request, book, "relancez l’analyse")
    if busy:
        return busy
    if not book.pages.exists():
        messages.error(request, "Importez des pages avant de lancer l’analyse.")
        return redirect("book_detail", pk=pk)
    try:
        jobs.start(book.pk, jobs.ANALYZE)
    except jobs.JobAlreadyRunning:
        messages.error(request, "Un traitement est déjà en cours sur ce livre.")
    else:
        messages.info(request, "Analyse lancée en arrière-plan : vous pouvez suivre son avancement ici.")
    return redirect("book_detail", pk=pk)

@login_required
def job_progress(request, pk):
    """Progress of the background job of a book, polled by the page."""
    book = get_object_or_404(Book, pk=pk, owner=request.user)
    return JsonResponse(jobs.state(book))

@login_required
@require_POST
def cancel_job(request, pk):
    book = get_object_or_404(Book, pk=pk, owner=request.user)
    jobs.cancel(book.pk)
    return JsonResponse(jobs.state(Book.objects.get(pk=book.pk)))
@login_required
@require_POST
def set_page_type(request, pk, page_pk):
    """Correct the type the analysis gave to a page ("" = not analysed). Prices follow the type."""
    book = jobs.recover_if_stale(get_object_or_404(Book, pk=pk, owner=request.user))
    busy = _refuse_json_while_busy(book)
    if busy:
        return busy
    page = get_object_or_404(PageImage, pk=page_pk, book=book)
    kind = request.POST.get("type", "")
    if kind and kind not in TYPES:
        return JsonResponse({"error": "Type inconnu."}, status=400)
    page.page_type = kind
    page.save(update_fields=["page_type"])
    return JsonResponse({"type": page.page_type, "icon": page.type_icon})

@login_required
@require_POST
def set_page_trim(request, pk, page_pk):
    """Give one page its own edge trim (or hand it back to the book's trim)."""
    book = jobs.recover_if_stale(get_object_or_404(Book, pk=pk, owner=request.user))
    busy = _refuse_json_while_busy(book)
    if busy:
        return busy
    page = get_object_or_404(PageImage, pk=page_pk, book=book)

    def percent(name):
        try:
            return min(max(float(request.POST.get(name, 0)), 0.0), 25.0)
        except ValueError:
            return 0.0

    page.trim_custom = request.POST.get("custom") == "1"
    page.trim_left, page.trim_top = percent("left"), percent("top")
    page.trim_right, page.trim_bottom = percent("right"), percent("bottom")
    page.save(update_fields=["trim_custom", "trim_left", "trim_top", "trim_right", "trim_bottom"])
    return JsonResponse({"custom": page.trim_custom, "left": page.trim_left, "top": page.trim_top,
                         "right": page.trim_right, "bottom": page.trim_bottom})

@login_required
@require_POST
def toggle_photo(request, pk, page_pk):
    """Mark or unmark a page as 'with photo' (it then uses the photo rendering settings)."""
    book = jobs.recover_if_stale(get_object_or_404(Book, pk=pk, owner=request.user))
    busy = _refuse_json_while_busy(book)
    if busy:
        return busy
    page = get_object_or_404(PageImage, pk=page_pk, book=book)
    page.has_photo = request.POST.get("value") == "1"
    page.save(update_fields=["has_photo"])
    return JsonResponse({"has_photo": page.has_photo})

@login_required
@require_POST
def save_settings(request, pk):
    book = jobs.recover_if_stale(get_object_or_404(Book, pk=pk, owner=request.user))
    busy = _refuse_while_busy(request, book, "enregistrez les réglages")
    if busy:
        return busy
    form = ProcessingSettingsForm(request.POST, instance=book)
    if form.is_valid():
        form.save()
        messages.success(request, "Paramètres enregistrés.")
    else:
        messages.error(request, "Certains paramètres sont invalides.")
    return redirect("book_detail", pk=pk)

@login_required
@require_POST
def process_book(request, pk):
    """Save the settings shown on the page, then build the searchable PDF in the background."""
    book = jobs.recover_if_stale(get_object_or_404(Book, pk=pk, owner=request.user))
    busy = _refuse_while_busy(request, book, "relancez le traitement")
    if busy:
        return busy
    form = ProcessingSettingsForm(request.POST, instance=book)
    if not form.is_valid():
        messages.error(request, "Paramètres de traitement invalides.")
        return redirect("book_detail", pk=pk)
    form.save()
    if not book.pages.exists():
        messages.error(request, "Importez les pages avant de lancer l’OCR.")
        return redirect("book_detail", pk=pk)
    try:
        jobs.start(book.pk, jobs.PROCESS)
    except jobs.JobAlreadyRunning:
        messages.error(request, "Un traitement est déjà en cours sur ce livre.")
    else:
        messages.info(request, "Traitement lancé en arrière-plan : vous pouvez quitter cette page, le PDF sera prêt à la fin.")
    return redirect("book_detail", pk=pk)
@login_required
def download_pdf(request, pk):
    book = get_object_or_404(Book, pk=pk, owner=request.user)
    if not book.output_pdf:
        raise Http404("Aucun PDF disponible.")
    return FileResponse(book.output_pdf.open("rb"), as_attachment=True, filename=f"{get_valid_filename(book.title) or 'livre'}.pdf")

@login_required
@require_POST
def delete_book(request, pk):
    book = jobs.recover_if_stale(get_object_or_404(Book, pk=pk, owner=request.user))
    busy = _refuse_while_busy(request, book, "supprimez le livre")
    if busy:
        return busy
    book.delete()
    messages.success(request, "Livre supprimé.")
    return redirect("dashboard")
