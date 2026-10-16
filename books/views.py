import io
import json
from pathlib import Path
import tempfile
from django.contrib import messages
from django.core.files.storage import default_storage
from django.contrib.auth.decorators import login_required
from django.db.models import Count, Q
from django.http import FileResponse, Http404, HttpResponse, JsonResponse
from PIL import Image, ImageOps
from .classify import CHART, PHOTO, TEXT, classify_page
from .image_processing import crop_to_page, process_image, same_side_neighbours, sibling_quad
from django.utils.text import get_valid_filename
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST
from .forms import BookCreateForm, ProcessingSettingsForm
from .models import Book, PageImage
from .ocr import build_searchable_pdf, OCRProcessingError

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
MAX_FILES_PER_UPLOAD = 500
MAX_FILE_SIZE = 30 * 1024 * 1024

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
    books = list(
        Book.objects.filter(owner=request.user)
        .annotate(page_count=Count("pages"))
    )
    summary = Book.objects.filter(owner=request.user).aggregate(
        book_count=Count("pk", distinct=True),
        page_count=Count("pages"),
        ready_count=Count("pk", filter=Q(status="done"), distinct=True),
    )
    return render(request, "books/dashboard.html", {
        "books": books,
        "summary": summary,
    })

@login_required
def create_book(request):
    if request.method == "POST":
        form = BookCreateForm(request.POST)
        if form.is_valid():
            book = form.save(commit=False)
            book.owner = request.user
            book.save()
            messages.success(request, "Livre créé. Vous pouvez maintenant importer les pages.")
            return redirect("book_detail", pk=book.pk)
    else:
        form = BookCreateForm()
    return render(request, "books/create_book.html", {"form": form})

@login_required
def book_detail(request, pk):
    book = get_object_or_404(Book, pk=pk, owner=request.user)
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
            "chart": sum(1 for p in pages if p.page_type == CHART),
            "photo": sum(1 for p in pages if p.page_type == PHOTO),
            "unknown": sum(1 for p in pages if not p.page_type),
        },
    })

@login_required
@require_POST
def upload_pages(request, pk):
    book = get_object_or_404(Book, pk=pk, owner=request.user)
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
        image = process_image(Path(page.image.path), draft, max_side=size, extra_rotation=page.rotation,
                              fallback_sources=neighbours, photo_page=page.has_photo, page_trim=page.custom_trim)
    except Exception:
        return HttpResponse("Impossible de générer l’aperçu.", status=500)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=85)
    response = HttpResponse(buffer.getvalue(), content_type="image/jpeg")
    response["Cache-Control"] = "no-store"
    return response

@login_required
@require_POST
def rotate_page(request, pk, page_pk):
    """Turn one page by 90 degrees, on top of the automatic rotation."""
    book = get_object_or_404(Book, pk=pk, owner=request.user)
    page = get_object_or_404(PageImage, pk=page_pk, book=book)
    step = -90 if request.POST.get("direction") == "ccw" else 90
    page.rotation = (page.rotation + step) % 360
    page.save(update_fields=["rotation"])
    return JsonResponse({"rotation": page.rotation})

@login_required
@require_POST
def analyze_pages(request, pk):
    """Detect the type of every page (text, graph/curve, image) and count them."""
    book = get_object_or_404(Book, pk=pk, owner=request.user)
    pages = list(book.pages.all())
    if not pages:
        messages.error(request, "Importez des pages avant de lancer l’analyse.")
        return redirect("book_detail", pk=pk)
    counts = {TEXT: 0, CHART: 0, PHOTO: 0}
    failed = 0
    for page in pages:
        try:
            with Image.open(page.image.path) as original:
                original.draft("RGB", (2000, 2000))   # fast partial decode: enough to tell page types apart
                photo = ImageOps.exif_transpose(original).convert("RGB")
            if book.auto_crop_page:
                neighbours = same_side_neighbours(pages, page)
                photo = crop_to_page(photo, lambda: sibling_quad(neighbours))
            kind, _ = classify_page(photo)
        except Exception:
            failed += 1
            continue
        counts[kind] += 1
        page.page_type = kind
        page.has_photo = kind == PHOTO     # image pages use the photo settings; change it per page if needed
        page.save(update_fields=["page_type", "has_photo"])
    messages.success(request, (
        f"Analyse terminée : {counts[TEXT]} page(s) de texte, {counts[CHART]} page(s) avec graphique ou courbe, "
        f"{counts[PHOTO]} page(s) avec image."))
    if failed:
        messages.warning(request, f"{failed} page(s) n’ont pas pu être analysées.")
    return redirect("book_detail", pk=pk)

@login_required
@require_POST
def set_page_trim(request, pk, page_pk):
    """Give one page its own edge trim (or hand it back to the book's trim)."""
    book = get_object_or_404(Book, pk=pk, owner=request.user)
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
    book = get_object_or_404(Book, pk=pk, owner=request.user)
    page = get_object_or_404(PageImage, pk=page_pk, book=book)
    page.has_photo = request.POST.get("value") == "1"
    page.save(update_fields=["has_photo"])
    return JsonResponse({"has_photo": page.has_photo})

@login_required
@require_POST
def save_settings(request, pk):
    book = get_object_or_404(Book, pk=pk, owner=request.user)
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
    book = get_object_or_404(Book, pk=pk, owner=request.user)
    form = ProcessingSettingsForm(request.POST, instance=book)
    if not form.is_valid():
        messages.error(request, "Paramètres de traitement invalides.")
        return redirect("book_detail", pk=pk)
    form.save()
    pages = list(book.pages.all())
    if not pages:
        messages.error(request, "Importez les pages avant de lancer l’OCR.")
        return redirect("book_detail", pk=pk)

    book.status = "processing"
    book.error_message = ""
    book.save(update_fields=["status", "error_message", "updated_at"])
    try:
        with tempfile.TemporaryDirectory(prefix="livreocr_export_") as tmp:
            output = Path(tmp) / "livre_recherchable.pdf"
            toc = build_searchable_pdf(book, pages, output)
            with output.open("rb") as pdf_file:
                filename = f"livre_{book.pk}_recherchable.pdf"
                book.output_pdf.save(filename, pdf_file, save=False)
            book.status = "done"
            book.save()
        messages.success(request, "PDF consultable créé avec succès.")
        if toc.get("links"):
            messages.success(request, (
                f"Table des matières : {toc['links']} lien(s) cliquable(s) sur {toc['toc_pages']} page(s)"
                + (f" ; {toc['unresolved']} entrée(s) renvoient à des pages absentes du livre et n’ont pas de lien." if toc["unresolved"] else ".")))
        elif toc.get("toc_pages"):
            messages.warning(request, "Une table des matières a été repérée, mais aucune de ses entrées ne renvoie à une page importée.")
    except Exception as exc:
        book.status = "error"
        book.error_message = str(exc)[:6000]
        book.save(update_fields=["status", "error_message", "updated_at"])
        messages.error(request, "Le traitement a échoué. Consultez le détail de l’erreur.")
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
    book = get_object_or_404(Book, pk=pk, owner=request.user)
    book.delete()
    messages.success(request, "Livre supprimé.")
    return redirect("dashboard")
