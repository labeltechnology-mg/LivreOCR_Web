from pathlib import Path
import shutil
import subprocess
import tempfile
from PIL import Image
from .image_processing import preprocess_image, same_side_neighbours

class OCRProcessingError(Exception):
    pass

def build_searchable_pdf(book, pages, output_path: Path) -> None:
    """Create a PDF from processed page images and add a searchable OCR text layer."""
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
            source = Path(page.image.path)
            target = tmpdir / f"page_{index:05d}.png"
            preprocess_image(source, target, book, page.rotation, same_side_neighbours(pages, page), page.has_photo)
            processed_paths.append(target)

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
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 or not output_path.exists():
            raise OCRProcessingError((result.stderr or result.stdout or "Échec OCR")[-6000:])
