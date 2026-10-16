import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from PIL import Image
from .image_processing import preprocess_image, same_side_neighbours
from .toc import add_toc_links

class OCRProcessingError(Exception):
    pass

def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60} min {seconds % 60:02d} s" if seconds >= 60 else f"{seconds} s"

def build_searchable_pdf(book, pages, output_path: Path, progress=None) -> dict:
    """Create a PDF from processed page images and add a searchable OCR text layer.

    progress(fraction, step) is called as the work advances; it may raise to stop the work (the
    OCR process is then killed). Returns the table-of-contents statistics ({} when links are off
    or nothing was found).
    """
    report = progress or (lambda fraction, step: None)
    if not shutil.which("ocrmypdf"):
        raise OCRProcessingError(
            "La commande ocrmypdf est introuvable. Installez OCRmyPDF et Tesseract."
        )
    if not pages:
        raise OCRProcessingError("Aucune page n’a été importée.")

    with tempfile.TemporaryDirectory(prefix="livreocr_") as tmp:
        tmpdir = Path(tmp)
        processed_paths = []
        for index, page in enumerate(pages, start=1):
            report(0.35 * (index - 1) / len(pages), f"Préparation de la page {index} sur {len(pages)}")
            source = Path(page.image.path)
            target = tmpdir / f"page_{index:05d}.png"
            preprocess_image(source, target, book, page.rotation, same_side_neighbours(pages, page), page.has_photo, page.custom_trim)
            processed_paths.append(target)

        report(0.35, "Assemblage des pages…")
        image_pdf = tmpdir / "images.pdf"
        with Image.open(processed_paths[0]) as first_source:
            first = first_source.convert("RGB")
        rest = []
        for path in processed_paths[1:]:
            with Image.open(path) as im:
                rest.append(im.convert("RGB").copy())
        first.save(image_pdf, "PDF", save_all=True, append_images=rest, resolution=300.0)
        first.close()
        for im in rest:
            im.close()

        command = ["ocrmypdf", "--output-type", "pdf", "-l", book.language, "-O", "1"]
        # Page rotation is handled in preprocess_image (auto-detection + manual per-page turns),
        # so the preview matches the final PDF and ocrmypdf must not rotate again.
        # Deskew is done in preprocess_image so the preview matches the final PDF.
        command += [str(image_pdf), str(output_path)]
        started = time.monotonic()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        expected = 10 + 3 * len(pages)          # a rough guess of the OCR time, in seconds: only shapes the bar
        try:
            while True:
                try:
                    stdout, stderr = process.communicate(timeout=1.5)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = time.monotonic() - started
                    # Creeps towards 0.95 and never reaches it: the OCR has no measurable progress of its own.
                    report(0.35 + 0.60 * (1 - math.exp(-elapsed / expected)),
                           f"Reconnaissance du texte (OCR)… {_format_duration(elapsed)}")
        except BaseException:                    # cancelled, or the server is stopping: do not leave the OCR running
            process.kill()
            process.communicate()
            raise
        if process.returncode != 0 or not output_path.exists():
            raise OCRProcessingError((stderr or stdout or "Échec OCR")[-6000:])

        if not book.toc_links:
            return {}
        report(0.95, "Liens de la table des matières…")
        try:
            # The page images are re-read here, so this must run before the temporary folder is removed.
            return add_toc_links(output_path, processed_paths, book.language)
        except Exception:
            return {}   # links are a bonus: never lose the finished PDF because of them
