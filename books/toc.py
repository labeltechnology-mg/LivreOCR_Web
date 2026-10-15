"""Make the table of contents of a book clickable in the finished PDF.

On a table of contents page each entry looks like "Title .......... 17": the number is the
*printed* page number of the target. We find which PDF page carries that printed number (read
from the page headers/footers), then add a link over the entry and a bookmark in the outline.

Tesseract's automatic page mode tends to drop the lone page numbers at the right of dotted
leaders, so the candidate pages are read a second time as one uniform block (psm 6), which
keeps them. That second reading is used only to find the entries, never to change the text layer.
"""
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import tempfile

import cv2
import numpy as np
import pikepdf
from PIL import Image, ImageOps
from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTFigure, LTTextContainer, LTTextLine

from .orientation import find_tesseract

HEADINGS = re.compile(r"\b(table\s+des\s+mati[eè]res|sommaire|inhaltsverzeichnis|inhalt|contents|indice|index)\b", re.I)
ENTRY = re.compile(r"^(?P<title>.*?\S)[\s.·…_\-–—\"'“”«»]*?(?P<number>\d{1,4})\s*$")
LEADERS = re.compile(r"(\.\s?){3,}|…|_{3,}|·{3,}")
PAGE_NUMBER = re.compile(r"^(?:p(?:age|\.)?\s*)?[|\[(]?\s*(\d{1,4})\s*[|\])]?$", re.I)
MAX_PRINTED_PAGE = 2000
MIN_ENTRIES = 4            # entries needed on one page to call it a table of contents
MIN_ENTRIES_WITH_HEADING = 3   # a "Sommaire" heading makes a shorter list believable
EDGE_BAND = 0.14           # a printed page number sits within this share of the top or bottom edge


MIN_BAND_CONFIDENCE = 70    # OCR confidence (0-100) required for a printed page number read in a band


@dataclass
class Line:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class Entry:
    title: str
    number: int
    line: Line


def _merge_rows(lines: list[Line]) -> list[Line]:
    """Join the pieces of one text row (a title far from its number) into a single line."""
    rows: list[list[Line]] = []
    for line in sorted(lines, key=lambda l: -(l.y0 + l.y1) / 2):
        centre = (line.y0 + line.y1) / 2
        for row in rows:
            ref = row[0]
            if abs(centre - (ref.y0 + ref.y1) / 2) < 0.5 * min(line.y1 - line.y0, ref.y1 - ref.y0):
                row.append(line)
                break
        else:
            rows.append([line])
    merged = []
    for row in rows:
        row.sort(key=lambda l: l.x0)
        merged.append(Line(" ".join(l.text for l in row), min(l.x0 for l in row), min(l.y0 for l in row),
                           max(l.x1 for l in row), max(l.y1 for l in row)))
    return merged


def _read_lines(pdf_path: Path):
    """Every text line of every page, with its position, from the OCR text layer."""
    def collect(node, lines):
        # OCR text layers sit inside figure objects, so the whole tree has to be walked.
        for child in node:
            if isinstance(child, LTTextLine):
                text = " ".join(child.get_text().split())
                if text:
                    lines.append(Line(text, child.x0, child.y0, child.x1, child.y1))
            elif isinstance(child, (LTTextContainer, LTFigure)):
                collect(child, lines)

    pages = []
    for layout in extract_pages(str(pdf_path), laparams=LAParams(line_margin=0.3, all_texts=True)):
        lines: list[Line] = []
        collect(layout, lines)
        pages.append((layout.width, layout.height, _merge_rows(lines)))
    return pages


NUMBER_COLUMN_WIDTH = 0.055   # width of the strip holding the page numbers, as a share of the page width
LEADER_JUNK = re.compile(r"(?:\s+[ilI1|.,:;'\"`~_\-]{1,6})+$")   # dot leaders misread as "iii", quotes...


def _tsv_words(tesseract: str, image: Image.Image, language: str, extra: list[str]) -> list[tuple]:
    """Words read by Tesseract as (left, top, width, height, text, line key), in pixels."""
    with tempfile.TemporaryDirectory(prefix="livreocr_read_") as tmp:
        path = Path(tmp) / "page.png"
        image.save(path)
        result = subprocess.run([tesseract, str(path), "stdout", "-l", language, "--psm", "6", *extra, "tsv"],
                                capture_output=True, text=True, timeout=180)
    words = []
    if result.returncode != 0:
        return words
    for raw in result.stdout.splitlines()[1:]:
        cols = raw.split("\t")
        if len(cols) < 12 or not cols[11].strip():
            continue
        try:
            left, top, w, h = (int(cols[i]) for i in (6, 7, 8, 9))
        except ValueError:
            continue
        words.append((left, top, w, h, cols[11].strip(), (cols[2], cols[3], cols[4])))
    return words


def remove_dot_leaders(image: Image.Image) -> Image.Image:
    """Erase the rows of dots that join a title to its page number.

    Used only on the copy that is read to find the entries: the PDF keeps its dots. Only long,
    evenly spaced rows of identical dots go, so punctuation, "i" dots and ellipses stay.
    """
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    width = gray.shape[1]
    ink = (gray < 0.6 * float(np.percentile(gray, 90))).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    lo, hi = max(2, 0.0012 * width), 0.0055 * width      # a dot of a leader is a few pixels wide
    dots = [i for i in range(1, count)
            if lo <= stats[i][2] <= hi and lo <= stats[i][3] <= hi
            and 0.6 <= stats[i][2] / stats[i][3] <= 1.7 and stats[i][4] >= 0.45 * stats[i][2] * stats[i][3]]
    if len(dots) < 6:
        return image
    dots.sort(key=lambda i: centroids[i][1])
    erase: list[int] = []
    rows: list[list[int]] = [[dots[0]]]
    for i in dots[1:]:
        if abs(centroids[i][1] - centroids[rows[-1][0]][1]) <= hi:
            rows[-1].append(i)
        else:
            rows.append([i])
    for row in rows:
        if len(row) < 6:
            continue
        row.sort(key=lambda j: centroids[j][0])
        xs = np.array([centroids[j][0] for j in row])
        gaps = np.diff(xs)
        typical = float(np.median(gaps))
        # Keep the dots that sit at the typical spacing of the row: stray marks ("i" dots,
        # commas) have irregular gaps and stay.
        regular = [j for k, j in enumerate(row)
                   if (k > 0 and 0.75 * typical <= gaps[k - 1] <= 1.25 * typical)
                   or (k < len(gaps) and 0.75 * typical <= gaps[k] <= 1.25 * typical)]
        if len(regular) >= 6:
            erase.extend(regular)
    if not erase:
        return image
    mask = cv2.dilate(np.isin(labels, erase).astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
    return Image.fromarray(cv2.inpaint(np.array(image.convert("RGB")), mask, 3, cv2.INPAINT_TELEA))


def _column_numbers(tesseract: str, strip: Image.Image) -> list[tuple]:
    """Read every number of a narrow column, one at a time.

    Page numbers in a table of contents are often small, condensed or stylised, and a whole
    column read at once merges or misreads them. Here each group of ink is cut out, enlarged
    and read alone as a single line of digits. Returns (left, top, width, height, digits, None).
    """
    gray = np.array(strip.convert("L"))
    ink = (gray < 0.6 * float(np.percentile(gray, 90))).astype(np.uint8)
    # Join the digits of one number, but keep numbers of different rows apart.
    joined = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, strip.width // 8), 1)))
    count, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    tokens = []
    pad = max(4, strip.height // 400)
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if h < strip.height * 0.004 or h > strip.height * 0.05 or w > strip.width * 0.97 or area < 12:
            continue                               # specks, rules and big pictures are not numbers
        if x + w < strip.width * 0.8:
            continue                               # page numbers are right-aligned: this is the tail of something else
        box = strip.crop((max(0, x - pad), max(0, y - pad), min(strip.width, x + w + pad), min(strip.height, y + h + pad)))
        text = _read_digits(tesseract, box)
        if text:
            tokens.append((int(x), int(y), int(w), int(h), text, None))
    return tokens


def _read_digits(tesseract: str, box: Image.Image) -> str:
    """The digits in a small crop, or "" when two different readings disagree.

    A wrong page number would send the reader to the wrong page, which is worse than no link,
    so a number is only trusted when it is read the same way twice.
    """
    gray = box.convert("L")
    # Reading 1: enlarged 3x, contrast stretched, read as a single line.
    first = ImageOps.autocontrast(gray.resize((gray.width * 3, gray.height * 3), Image.LANCZOS))
    # Reading 2: enlarged 4x, hard black and white, read as a single word.
    second = gray.resize((gray.width * 4, gray.height * 4), Image.BICUBIC)
    level = float(np.percentile(np.array(second), 50))
    second = second.point(lambda v: 255 if v > 0.6 * level else 0)
    # Reading 3: no digit whitelist. Forcing digits makes a narrow stylised "4" come out as "0";
    # free reading lets the engine say it is not a digit, which exposes such errors.
    readings = []
    for image, psm, whitelist in ((first, "7", True), (second, "8", True), (first, "7", False)):
        with tempfile.TemporaryDirectory(prefix="livreocr_digit_") as tmp:
            path = Path(tmp) / "n.png"
            ImageOps.expand(image, border=24, fill=255).save(path)   # a white margin: Tesseract needs room around text
            command = [tesseract, str(path), "stdout", "-l", "eng", "--psm", psm]
            if whitelist:
                command += ["-c", "tessedit_char_whitelist=0123456789"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        readings.append("".join(result.stdout.split()) if result.returncode == 0 else "")
    return readings[0] if readings[0].isdigit() and readings[0] == readings[1] == readings[2] else ""


def _read_block(image_path: Path, language: str, width_pt: float, height_pt: float) -> list[Line]:
    """Read a page as one uniform block (psm 6) and return its lines in PDF coordinates.

    The page numbers are read separately, from the right-hand column alone and digits only: next
    to dot leaders Tesseract misreads or drops them, but a column of digits is easy.
    """
    tesseract = find_tesseract()
    if tesseract is None:
        return []
    with Image.open(image_path) as image:
        image = remove_dot_leaders(image.convert("RGB"))
        width_px, height_px = image.size
        body = _tsv_words(tesseract, image, language, [])
        # The page numbers are right-aligned: the right edge of the words seen in the last
        # column (the 90th percentile of all right edges) locates that column, and a narrow strip
        # ending there holds the page numbers alone, without any chapter number before them.
        rights = sorted(w[0] + w[2] for w in body)
        right_edge = rights[int(0.9 * (len(rights) - 1))] if len(rights) >= 6 else int(0.9 * width_px)
        column_x = int(max(width_px * 0.5, right_edge - width_px * NUMBER_COLUMN_WIDTH))
        # A margin on the right: the last digit of a number must not be cut off (a clipped "4" is read as "0").
        column_end = min(width_px, int(right_edge + width_px * 0.04))
        digits = _column_numbers(tesseract, image.crop((column_x, 0, column_end, height_px)))
    sx, sy = width_pt / width_px, height_pt / height_px

    rows: dict[tuple, list[tuple]] = {}
    for word in body:
        rows.setdefault(word[5], []).append(word)
    text_rows = []
    for words in rows.values():
        words.sort()
        text_rows.append((min(w[1] for w in words), max(w[1] + w[3] for w in words), words))

    lines = []
    used = set()
    # One entry per number of the right column: it is the most reliable thing on the page.
    # Its title is the text row at the same height, or a generic one when that row was not read.
    for number in sorted(digits, key=lambda d: d[1]):
        centre = number[1] + number[3] / 2
        match = next((i for i, (top, bottom, _) in enumerate(text_rows)
                      if top - 0.3 * (bottom - top) <= centre <= bottom + 0.3 * (bottom - top)), None)
        if match is not None:
            used.add(match)
            top, bottom, words = text_rows[match]
            title_words = [w for w in words if w[0] + w[2] <= column_x]
            title = LEADER_JUNK.sub("", " ".join(w[4] for w in title_words)).strip()
            left = min(w[0] for w in words)
        else:
            top, bottom = number[1], number[1] + number[3]
            title, left = "", column_x
        right = column_x + number[0] + number[2]
        lines.append(Line(f"{title or 'Page ' + number[4]} {number[4]}" if not title else f"{title} {number[4]}",
                          left * sx, height_pt - bottom * sy, right * sx, height_pt - top * sy))
    for index, (top, bottom, words) in enumerate(text_rows):      # the other rows, as read (headings...)
        if index not in used:
            lines.append(Line(" ".join(w[4] for w in words), min(w[0] for w in words) * sx, height_pt - bottom * sy,
                              max(w[0] + w[2] for w in words) * sx, height_pt - top * sy))
    return lines


def _numbers_in(text: str) -> list[int]:
    """Page-number-like tokens of a line: "16 |", "| 17", "page 35" or a lone "9"."""
    numbers = []
    for piece in re.split(r"\s{2,}|\s(?=[|\[(]?\d{1,4}[|\])]?$)", text):
        match = PAGE_NUMBER.match(piece.strip())
        if match and int(match.group(1)) <= MAX_PRINTED_PAGE:
            numbers.append(int(match.group(1)))
    return numbers


def _band_numbers(image_path: Path, language: str) -> list[int]:
    """Read the top and bottom bands of a page image on their own: lone page numbers are often
    dropped by the OCR of the whole page, but read fine when they are all there is to read."""
    tesseract = find_tesseract()
    if tesseract is None:
        return []
    numbers: list[int] = []
    with Image.open(image_path) as image:
        width, height = image.size
        bands = (image.crop((0, 0, width, int(height * EDGE_BAND))), image.crop((0, int(height * (1 - EDGE_BAND)), width, height)))
        for band in bands:
            with tempfile.TemporaryDirectory(prefix="livreocr_band_") as tmp:
                path = Path(tmp) / "band.png"
                band.save(path)
                result = subprocess.run([tesseract, str(path), "stdout", "-l", language, "--psm", "6", "tsv"],
                                        capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                continue
            for raw in result.stdout.splitlines()[1:]:
                cols = raw.split("\t")
                if len(cols) < 12:
                    continue
                try:
                    confident = float(cols[10]) >= MIN_BAND_CONFIDENCE
                except ValueError:
                    continue
                if confident:
                    numbers.extend(_numbers_in(cols[11].strip()))
    return numbers


def printed_page_numbers(pages, images: list[Path] | None = None, language: str = "fra+eng") -> dict[int, int]:
    """Printed page number -> PDF page index, read from lone numbers near the top/bottom edge."""
    found: dict[int, int] = {}
    for index, (width, height, lines) in enumerate(pages):
        candidates = []
        for line in lines:
            if line.y0 > height * (1 - EDGE_BAND) or line.y1 < height * EDGE_BAND:
                candidates.extend(_numbers_in(line.text))
        if not candidates and images and index < len(images):
            candidates = _band_numbers(images[index], language)
        if len(set(candidates)) == 1:        # ambiguous pages are ignored rather than guessed
            found.setdefault(candidates[0], index)
    return found


def _letters(text: str) -> int:
    return sum(ch.isalpha() for ch in text)


def _drop_misordered(entries: list[Entry]) -> list[Entry]:
    """Entries of a table of contents run in increasing page order. When they mostly do, an
    entry that goes backwards is almost surely a misread digit: drop it rather than link it."""
    if len(entries) < 4:
        return entries
    ordered = sum(b.number >= a.number for a, b in zip(entries, entries[1:]))
    if ordered < 0.7 * (len(entries) - 1):
        return entries                      # not a sorted list (e.g. an alphabetical index): keep as read
    kept: list[Entry] = []
    for entry in sorted_by_position(entries):
        if not kept or entry.number >= kept[-1].number:
            kept.append(entry)
    return kept


def sorted_by_position(entries: list[Entry]) -> list[Entry]:
    return sorted(entries, key=lambda e: -e.line.y1)   # top of the page first


def toc_entries(width: float, lines: list[Line]) -> tuple[list[Entry], bool]:
    """Entries of a page that looks like a table of contents, and whether it has a heading."""
    entries = []
    leaders = 0
    for line in lines:
        match = ENTRY.match(line.text)
        if not match:
            continue
        title = match.group("title").strip(" .·…_-–—")
        number = int(match.group("number"))
        # A title is words: this rejects table rows made of figures ("1887  1'325'000  89'000").
        if number > MAX_PRINTED_PAGE or _letters(title) < 3 or _letters(title) < 0.5 * len(title.replace(" ", "")):
            continue
        has_leader = bool(LEADERS.search(line.text))
        if not has_leader and line.x1 < width * 0.6:
            continue                            # short line: not a right-aligned page number
        leaders += has_leader
        entries.append(Entry(title, number, line))
    entries = _drop_misordered(entries)
    heading = any(HEADINGS.search(line.text) for line in lines)
    if len(entries) < (MIN_ENTRIES_WITH_HEADING if heading else MIN_ENTRIES):
        return [], heading
    if not heading and leaders < 0.6 * len(entries):
        return [], heading                      # ordinary text that happens to end with numbers
    return entries, heading


def _is_candidate(lines: list[Line]) -> bool:
    """Cheap test on the normal text layer: a heading, or several rows with dot leaders."""
    return any(HEADINGS.search(l.text) for l in lines) or sum(bool(LEADERS.search(l.text)) for l in lines) >= 3


def _offset(found: dict[int, int]) -> int | None:
    """Constant (PDF index - printed number) when the pages found are consecutive, else None."""
    if len(found) < 3:
        return None
    offsets = [index - number for number, index in found.items()]
    return offsets[0] if len(set(offsets)) == 1 else None


def add_toc_links(pdf_path: Path, images: list[Path] | None = None, language: str = "fra+eng") -> dict:
    """Add links and bookmarks for the table(s) of contents of the PDF, in place.

    images: the page images the PDF was built from (same order), used to re-read candidate
    pages. Returns {"toc_pages": n, "links": n, "unresolved": n}.
    """
    pages = _read_lines(pdf_path)
    stats = {"toc_pages": 0, "links": 0, "unresolved": 0}
    found: list[tuple[int, list[Entry]]] = []
    for index, (width, height, lines) in enumerate(pages):
        if not _is_candidate(lines):
            continue
        entries, _ = toc_entries(width, lines)
        if images and index < len(images):
            block_entries, _ = toc_entries(width, _read_block(images[index], language, width, height))
            if len(block_entries) >= len(entries):
                entries = block_entries         # the uniform-block reading keeps the page numbers
        if entries:
            found.append((index, entries))
    if not found:
        return stats                            # no table of contents: skip the page-number reading

    printed = printed_page_numbers(pages, images, language)
    offset = _offset(printed)
    plans = []                                  # (pdf page index, entry, target index)
    for index, entries in found:
        stats["toc_pages"] += 1
        for entry in entries:
            target = printed.get(entry.number)
            if target is None and offset is not None and 0 <= entry.number + offset < len(pages):
                target = entry.number + offset  # pages are consecutive, so the number maps directly
            if target is None or target == index:
                stats["unresolved"] += 1        # target page is not in this book, or unreadable
                continue
            plans.append((index, entry, target))
    if not plans:
        return stats

    with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
        for index, entry, target in plans:
            page = pdf.pages[index]
            line = entry.line
            link = pdf.make_indirect(pikepdf.Dictionary(
                Type=pikepdf.Name.Annot, Subtype=pikepdf.Name.Link,
                Rect=[line.x0 - 2, line.y0 - 2, line.x1 + 2, line.y1 + 2],
                Border=[0, 0, 0],
                Dest=pikepdf.Array([pdf.pages[target].obj, pikepdf.Name.Fit]),
            ))
            if "/Annots" not in page:
                page.Annots = pikepdf.Array()
            page.Annots.append(link)
            stats["links"] += 1
        with pdf.open_outline() as outline:
            if not len(outline.root):
                for index, entry, target in plans:
                    outline.root.append(pikepdf.OutlineItem(entry.title[:120], target))
        pdf.save(pdf_path)
    return stats
