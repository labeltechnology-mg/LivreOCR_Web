"""Long operations (OCR of a book, analysis of its pages) run in the background.

No extra service is needed: each job is a thread of the web process. Its progress, its
cancellation flag and a heartbeat live in the database, so any web worker can show and cancel
it, and a job that died with its process (restart, crash) is noticed by its stale heartbeat.
"""
from pathlib import Path
import tempfile
import threading
import time

from django.conf import settings
from django.db import close_old_connections, connection
from django.utils import timezone
from PIL import Image, ImageOps

from .classify import CHART, PHOTO, TABLE, TEXT, classify_page
from .image_processing import crop_to_page, same_side_neighbours, sibling_quad
from .models import Book
from .ocr import build_searchable_pdf

STALE_AFTER = 180          # seconds without a heartbeat after which a job is considered dead
PROCESS, ANALYZE = "process", "analyze"
TITLES = {PROCESS: "Création du PDF consultable", ANALYZE: "Analyse des pages"}

_threads: dict[int, threading.Thread] = {}
_lock = threading.Lock()


class JobCancelled(Exception):
    pass


class JobAlreadyRunning(Exception):
    pass


def _update(book_id: int, **fields) -> None:
    Book.objects.filter(pk=book_id).update(**fields)


class Progress:
    """Handed to the worker code: records progress, beats, and raises JobCancelled when asked to stop."""

    def __init__(self, book_id: int):
        self.book_id = book_id
        self._last_write = 0.0

    def __call__(self, fraction: float, step: str) -> None:
        if Book.objects.filter(pk=self.book_id, job_cancel=True).exists():
            raise JobCancelled()
        now = time.monotonic()
        if now - self._last_write < 0.4 and fraction < 1:
            return                                   # no need to write more often than the page polls
        self._last_write = now
        _update(self.book_id, job_progress=max(0, min(1000, int(fraction * 1000))), job_step=step[:200],
                job_heartbeat=timezone.now())


def is_active(book: Book) -> bool:
    return bool(book.job_kind)


def start(book_id: int, kind: str) -> None:
    """Mark the book as busy and run the job (in a thread, or inline when JOBS_RUN_INLINE is set)."""
    with _lock:
        fields = dict(job_kind=kind, job_progress=0, job_step="Démarrage…", job_heartbeat=timezone.now(),
                      job_cancel=False, job_result="")
        if kind == PROCESS:
            fields.update(status="processing", error_message="")
        claimed = Book.objects.filter(pk=book_id, job_kind="").update(**fields)
        if not claimed:
            if not Book.objects.filter(pk=book_id).exists():
                raise Book.DoesNotExist(f"Book {book_id} does not exist")
            raise JobAlreadyRunning()
    if getattr(settings, "JOBS_RUN_INLINE", False):
        _run(book_id, kind)
        return
    thread = threading.Thread(target=_run, args=(book_id, kind), name=f"job-{kind}-{book_id}", daemon=True)
    _threads[book_id] = thread
    thread.start()


def cancel(book_id: int) -> None:
    Book.objects.filter(pk=book_id).exclude(job_kind="").update(job_cancel=True, job_step="Annulation en cours…")


def recover_if_stale(book: Book) -> Book:
    """A job whose worker disappeared (server restarted...) is closed with an explanation."""
    if not book.job_kind:
        return book
    beat = book.job_heartbeat
    if beat is not None and (timezone.now() - beat).total_seconds() < STALE_AFTER:
        return book
    stale_job = Book.objects.filter(pk=book.pk, job_kind=book.job_kind, job_heartbeat=beat)
    fields = dict(job_kind="", job_cancel=False, job_heartbeat=None, job_step="",
                  job_result="error|Le traitement a été interrompu (redémarrage du serveur ?). Relancez-le.")
    if book.status == "processing":
        fields.update(status="error", error_message="Traitement interrompu avant la fin. Relancez-le.")
    stale_job.update(**fields)
    book.refresh_from_db()
    return book


def state(book: Book) -> dict:
    book = recover_if_stale(book)
    return {"active": bool(book.job_kind), "kind": book.job_kind, "title": TITLES.get(book.job_kind, ""),
            "percent": round(book.job_progress / 10, 1), "step": book.job_step, "status": book.status,
            "cancelling": book.job_cancel}


def take_result(book: Book) -> list[tuple[str, str]]:
    """The messages left by the last finished job, as (level, text); they are shown once."""
    if book.job_kind or not book.job_result:
        return []
    lines = [tuple(line.split("|", 1)) for line in book.job_result.splitlines() if "|" in line]
    _update(book.pk, job_result="")
    return lines


def _run(book_id: int, kind: str) -> None:
    close_old_connections()
    progress = Progress(book_id)
    try:
        book = Book.objects.get(pk=book_id)
        result = _process(book, progress) if kind == PROCESS else _analyze(book, progress)
        _update(book_id, job_result=result)
    except JobCancelled:
        fields = dict(job_result="warning|Opération annulée.")
        if kind == PROCESS:
            fields.update(status="new")
        _update(book_id, **fields)
    except Exception as exc:                                    # shown to the user, never lost in a thread
        fields = dict(job_result=f"error|L’opération a échoué : {str(exc)[:300]}")
        if kind == PROCESS:
            fields.update(status="error", error_message=str(exc)[:6000])
        _update(book_id, **fields)
    finally:
        _update(book_id, job_kind="", job_cancel=False, job_heartbeat=None, job_step="", job_progress=0)
        _threads.pop(book_id, None)
        if not getattr(settings, "JOBS_RUN_INLINE", False):
            connection.close()                                  # this thread's own connection


def _process(book: Book, progress: Progress) -> str:
    pages = list(book.pages.all())
    if not pages:
        raise ValueError("Aucune page n’a été importée.")
    with tempfile.TemporaryDirectory(prefix="livreocr_export_") as tmp:
        output = Path(tmp) / "livre_recherchable.pdf"
        toc = build_searchable_pdf(book, pages, output, progress=progress)
        progress(0.97, "Enregistrement du PDF…")
        with output.open("rb") as pdf_file:
            book.output_pdf.save(f"livre_{book.pk}_recherchable.pdf", pdf_file, save=False)
    _update(book.pk, output_pdf=book.output_pdf.name, status="done")
    lines = ["success|PDF consultable créé avec succès."]
    if toc.get("links"):
        lines.append(
            f"success|Table des matières : {toc['links']} lien(s) cliquable(s) sur {toc['toc_pages']} page(s)"
            + (f" ; {toc['unresolved']} entrée(s) renvoient à des pages absentes du livre et n’ont pas de lien."
               if toc["unresolved"] else "."))
    elif toc.get("toc_pages"):
        lines.append("warning|Une table des matières a été repérée, mais aucune de ses entrées ne renvoie à une page importée.")
    return "\n".join(lines)


def _analyze(book: Book, progress: Progress) -> str:
    pages = list(book.pages.all())
    if not pages:
        raise ValueError("Importez des pages avant de lancer l’analyse.")
    counts = {TEXT: 0, TABLE: 0, CHART: 0, PHOTO: 0}
    failed = 0
    for index, page in enumerate(pages):
        progress(index / len(pages), f"Page {index + 1} sur {len(pages)}")
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
        # Image pages use the photo settings; this can be changed page by page afterwards.
        type(page).objects.filter(pk=page.pk).update(page_type=kind, has_photo=kind == PHOTO)
    lines = [(f"success|Analyse terminée : {counts[TEXT]} page(s) de texte seulement, {counts[TABLE]} avec tableau, "
              f"{counts[CHART]} avec graphique ou courbe, {counts[PHOTO]} avec image.")]
    if failed:
        lines.append(f"warning|{failed} page(s) n’ont pas pu être analysées.")
    return "\n".join(lines)
