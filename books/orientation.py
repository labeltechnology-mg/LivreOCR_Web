"""Detect pages that were photographed sideways or upside down, using Tesseract."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from PIL import Image, ImageOps

OSD_MIN_CONFIDENCE = 2.0       # below this OSD is ignored even as a last resort
OSD_TRUSTED_CONFIDENCE = 10.0  # at or above this OSD is accepted without a second check
SAMPLE_SIDE = 1500         # long side, in pixels, of the copy sent to Tesseract
_ROTATIONS = (0, 90, 180, 270)

def find_tesseract() -> str | None:
    found = shutil.which("tesseract")
    if found:
        return found
    default = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    return str(default) if default.exists() else None

def rotate_clockwise(image: Image.Image, degrees: int) -> Image.Image:
    degrees %= 360
    if degrees == 0:
        return image
    method = {90: Image.ROTATE_270, 180: Image.ROTATE_180, 270: Image.ROTATE_90}[degrees]
    return image.transpose(method)

def _run(tesseract: str, image: Image.Image, args: list[str], timeout: int = 90) -> str:
    with tempfile.TemporaryDirectory(prefix="livreocr_osd_") as tmp:
        path = Path(tmp) / "sample.png"
        image.save(path)
        result = subprocess.run([tesseract, str(path), "stdout", *args], capture_output=True, text=True, timeout=timeout)
    return result.stdout if result.returncode == 0 else ""

def _osd(tesseract: str, sample: Image.Image) -> tuple[int, float] | None:
    out = _run(tesseract, sample, ["--psm", "0"])
    rotate = re.search(r"Rotate:\s*(\d+)", out)
    confidence = re.search(r"Orientation confidence:\s*([\d.]+)", out)
    if not rotate or not confidence:
        return None
    return int(rotate.group(1)) % 360, float(confidence.group(1))

def _reading_score(tesseract: str, sample: Image.Image, language: str) -> float:
    """How much real text Tesseract reads: upright pages score far higher than rotated ones."""
    out = _run(tesseract, sample, ["-l", language, "--psm", "6", "tsv"])
    score = 0.0
    for line in out.splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) < 12:
            continue
        try:
            confidence = float(cols[10])
        except ValueError:
            continue
        word = cols[11].strip()
        if confidence >= 50 and len(re.findall(r"[^\W\d_]", word)) >= 3:
            score += confidence
    return score

def detect_rotation(image: Image.Image, language: str = "fra+eng") -> int:
    """Clockwise degrees (0, 90, 180 or 270) to turn the page so its text reads upright."""
    tesseract = find_tesseract()
    if tesseract is None:
        return 0
    sample = ImageOps.grayscale(image)
    sample.thumbnail((SAMPLE_SIDE, SAMPLE_SIDE))
    scores: dict[int, float] = {}

    def score(angle: int) -> float:
        if angle not in scores:
            scores[angle] = _reading_score(tesseract, rotate_clockwise(sample, angle), language)
        return scores[angle]

    try:
        osd = _osd(tesseract, sample)
        if osd and osd[1] >= OSD_TRUSTED_CONFIDENCE:
            return osd[0]
        # OSD is unsure (or wrong on pages dominated by a picture): check by actually reading.
        # Start with its answer and the opposite turn, the mistake it makes most often.
        guess = osd[0] if osd else 0
        opposite = (guess + 180) % 360
        if score(guess) >= 150 and score(guess) >= 1.5 * score(opposite):
            return guess
        if score(opposite) >= 150 and score(opposite) >= 1.5 * score(guess):
            return opposite   # OSD had the right axis but the wrong end: no need to test the others
        for angle in _ROTATIONS:
            score(angle)
    except (subprocess.TimeoutExpired, OSError):
        return 0
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    (best, best_score), (_, second_score) = ranked[0], ranked[1]
    if best_score >= 150 and best_score >= 1.5 * second_score:
        return best
    # Nothing readable: fall back on OSD when it had at least some confidence.
    return osd[0] if osd and osd[1] >= OSD_MIN_CONFIDENCE else 0
