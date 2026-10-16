"""Tell text pages, tables, graph/curve pages and pages with an image apart."""
import cv2
import numpy as np
from PIL import Image

TEXT, TABLE, CHART, PHOTO = "text", "table", "chart", "photo"
TYPES = (TEXT, TABLE, CHART, PHOTO)
LABELS = {TEXT: "Texte seulement", TABLE: "Avec tableau", CHART: "Avec graphique / courbe", PHOTO: "Avec image"}
ICONS = {TEXT: "📄", TABLE: "▦", CHART: "📈", PHOTO: "🖼"}

WORK_SIDE = 1000          # the page is analysed on a copy this many pixels on its long side
PHOTO_MIN_AREA = 0.05     # share of the page a picture must cover to count as an image
PHOTO_MIN_TONE_RANGE = 55 # a picture spans many tones; a flat coloured box or a black bar does not
CHART_MIN_BLACK = 0.025   # share of the page covered by large solid black shapes (bars, plates)
CHART_MIN_DRAWING = 0.15  # share of the page spanned by one connected line drawing (curve, axes)
CHART_MAX_LETTERS = 450   # a drawing with more letters than this is a table or a text page
TABLE_MIN_H_RULES = 4     # long horizontal rules needed to call a page a table...
TABLE_MIN_V_RULES = 1     # ...and a vertical one (photographed thin column lines are often broken up)


def _prepare(page: Image.Image):
    """Grayscale copy of the page, scaled so that its paper is about 255."""
    scale = WORK_SIDE / max(page.size)
    small = page.resize((max(1, round(page.width * scale)), max(1, round(page.height * scale))), Image.BILINEAR) if scale < 1 else page
    gray = cv2.cvtColor(np.array(small), cv2.COLOR_RGB2GRAY).astype(np.float32)
    paper = max(float(np.percentile(gray, 90)), 1.0)
    return np.clip(gray * 255.0 / paper, 0, 255)


def _picture_area(norm: np.ndarray) -> float:
    """Share of the page covered by the largest picture (photo or illustration).

    Letters and thin lines are removed with a large morphological opening, which leaves the
    big non-white areas. Among them, only those with a wide range of tones are pictures:
    a coloured text box has a flat tone, a bar of a chart is uniformly black.
    """
    h, w = norm.shape
    size = max(15, w // 40) | 1
    non_paper = (norm < 0.88 * 255).astype(np.uint8)
    big = cv2.morphologyEx(non_paper, cv2.MORPH_OPEN, np.ones((size, size), np.uint8))
    smooth = cv2.medianBlur(norm.astype(np.uint8), 9)   # also erases text lying on the area
    count, labels, stats, _ = cv2.connectedComponentsWithStats(big, connectivity=8)
    best = 0.0
    for i in range(1, count):
        area = stats[i][4] / float(h * w)
        if area < PHOTO_MIN_AREA or area <= best:
            continue
        tones = smooth[labels == i]
        if np.percentile(tones, 90) - np.percentile(tones, 10) >= PHOTO_MIN_TONE_RANGE:
            best = area
    return best


def _chart_features(norm: np.ndarray) -> dict:
    """Features of line art: solid black shapes and long lines versus the amount of text."""
    ink = (norm < 110).astype(np.uint8)
    h, w = ink.shape
    # "Solid" means really black (bars of a chart), not a dark coloured box.
    solid = cv2.erode((norm < 70).astype(np.uint8), np.ones((9, 9), np.uint8))
    solid_share = float(cv2.dilate(solid, np.ones((9, 9), np.uint8)).mean())

    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 6), 1)))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, h // 8))))
    lines = cv2.bitwise_or(horizontal, vertical)

    # Letters are small connected shapes; everything left after removing lines and solids.
    rest = cv2.bitwise_and(ink, cv2.bitwise_not(cv2.dilate(lines, np.ones((3, 3), np.uint8))))
    rest = cv2.bitwise_and(rest, cv2.bitwise_not(cv2.dilate(solid, np.ones((9, 9), np.uint8))))
    count, _, stats, _ = cv2.connectedComponentsWithStats(rest, connectivity=8)
    letters = sum(1 for i in range(1, count) if 3 <= stats[i][2] <= 40 and 3 <= stats[i][3] <= 40 and stats[i][4] >= 6)

    # A drawing is a big connected structure of thin strokes: a curve, an axis frame, a grid.
    frame = cv2.dilate(ink, np.ones((5, 5), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(frame, connectivity=8)
    drawing = 0.0
    for i in range(1, count):
        bw, bh, area = stats[i][2], stats[i][3], stats[i][4]
        if bw * bh > 0.12 * h * w and area / float(bw * bh) < 0.35:
            drawing = max(drawing, bw * bh / float(h * w))
    # A table is a grid: several long horizontal rules crossed by several long vertical ones.
    def long_lines(mask, along_width, minimum):
        count, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(mask, np.ones((3, 3), np.uint8)), connectivity=8)
        return sum(1 for i in range(1, count) if (stats[i][2] if along_width else stats[i][3]) >= minimum)
    horizontal_rules = long_lines(horizontal, True, 0.25 * w)
    vertical_rules = long_lines(vertical, False, 0.06 * h)
    return {"solid": solid_share, "letters": letters, "drawing": drawing,
            "h_rules": horizontal_rules, "v_rules": vertical_rules}


def classify_page(page: Image.Image) -> tuple[str, dict]:
    """Return (type, features). The page should already be cut out and straightened.

    A page that contains a picture counts as an image page even if it also has text.
    A grid of rules around a lot of text is a table; a drawing with little text is a chart.
    """
    norm = _prepare(page)
    picture = _picture_area(norm)
    chart = _chart_features(norm)
    features = {"picture": round(picture, 3), "solid": round(chart["solid"], 3), "letters": chart["letters"],
                "drawing": round(chart["drawing"], 3), "h_rules": chart["h_rules"], "v_rules": chart["v_rules"]}
    if picture > 0:
        return PHOTO, features
    if chart["solid"] >= CHART_MIN_BLACK:
        return CHART, features
    if chart["h_rules"] >= TABLE_MIN_H_RULES and chart["v_rules"] >= TABLE_MIN_V_RULES:
        return TABLE, features
    if chart["drawing"] >= CHART_MIN_DRAWING and chart["letters"] < CHART_MAX_LETTERS:
        return CHART, features
    return TEXT, features
