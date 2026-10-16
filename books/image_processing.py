from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from .orientation import detect_rotation, rotate_clockwise

def crop_page(image: Image.Image, padding: int = 20) -> Image.Image:
    """Crop non-white outer margins conservatively; inspect results before archival use."""
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray < 245).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)
    # Avoid aggressive crop for sparse, mostly blank or very light pages.
    if w < image.width * 0.25 or h < image.height * 0.25:
        return image
    x0, y0 = max(0, x-padding), max(0, y-padding)
    x1, y1 = min(image.width, x+w+padding), min(image.height, y+h+padding)
    return image.crop((x0, y0, x1, y1))

CLAHE_CLIP_LIMIT = 1.45    # local contrast: moderate, so grain and paper texture are not amplified
BILATERAL_DIAMETER = 5     # edge-preserving smoothing
NLM_STRENGTH = 2.5         # non-local-means noise reduction ("h")
UNSHARP_SIGMA = 0.65       # blur radius of the unsharp mask, in pixels
MAX_PAPER_GAIN = 2.2  # never brighten a page by more than this when "whitening" its paper

def _paper_level(values: np.ndarray) -> float:
    """Brightness of the paper. A page that is mostly a dark picture has no real paper to
    measure, so the level is floored: otherwise such a page would be blown out."""
    return max(float(np.percentile(values, 90)), 255.0 / MAX_PAPER_GAIN, 1.0)

def _odd(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 else value + 1

def clean_background(image: Image.Image) -> Image.Image:
    """Flatten uneven lighting and shadows (book gutter, yellowed paper) to a white page."""
    rgb = np.array(image)
    small_scale = 4 if max(rgb.shape[:2]) > 1600 else 1
    channels = []
    for ch in cv2.split(rgb):
        small = cv2.resize(ch, None, fx=1 / small_scale, fy=1 / small_scale, interpolation=cv2.INTER_AREA) if small_scale > 1 else ch
        # Closing removes dark text so only the paper background remains.
        small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        bg = cv2.medianBlur(small, _odd(max(small.shape[:2]) // 25))
        # Large solid black areas (charts, plates) look like "background" too: never treat
        # anything darker than half the paper level as lighting, or they would be bleached.
        paper = _paper_level(small)
        bg = np.maximum(bg, int(0.5 * paper)).astype(np.uint8)
        if small_scale > 1:
            bg = cv2.resize(bg, (ch.shape[1], ch.shape[0]), interpolation=cv2.INTER_LINEAR)
        local = ch.astype(np.float32) * 255.0 / np.maximum(bg, 1)   # lighting-corrected value
        flat = ch.astype(np.float32) * 255.0 / max(paper, 1.0)      # one uniform gain for the whole page
        # Paper (and shadowed paper) gets the local correction; real black stays black.
        weight = np.clip((ch.astype(np.float32) / max(paper, 1.0) - 0.25) / 0.25, 0, 1)
        channels.append(np.clip(weight * local + (1 - weight) * flat, 0, 255).astype(np.uint8))
    return Image.fromarray(cv2.merge(channels))

def _order_points(points: np.ndarray) -> np.ndarray:
    """Return corners as top-left, top-right, bottom-right, bottom-left."""
    s = points.sum(axis=1)
    d = np.diff(points, axis=1).ravel()
    return np.array([points[np.argmin(s)], points[np.argmin(d)], points[np.argmax(s)], points[np.argmax(d)]], np.float32)

def detect_page_quad(image: Image.Image) -> np.ndarray | None:
    """Find the sheet of paper in a photo (bright page on a dark stand) and return its 4 corners.

    The page is the biggest bright blob that does not touch the picture border; the glass
    reflection under the page touches the border and is separated from it by a dark seam.
    Returns None when nothing page-like is found, so the caller keeps the whole image.
    """
    full_w, full_h = image.size
    scale = 1000 / max(full_w, full_h)
    small = image.resize((max(1, round(full_w * scale)), max(1, round(full_h * scale))), Image.BILINEAR) if scale < 1 else image
    scale = small.width / full_w
    gray = cv2.GaussianBlur(cv2.cvtColor(np.array(small), cv2.COLOR_RGB2GRAY), (5, 5), 0)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = gray[gray > otsu]
    if bright.size < gray.size * 0.1:
        return None
    # A second, stricter threshold cuts the dark seam between the page and its reflection.
    strict = max(otsu, 0.85 * float(np.median(bright)))
    # A min filter (not a blur) keeps thin dark seams from being averaged away.
    seam_safe = cv2.erode(cv2.cvtColor(np.array(small), cv2.COLOR_RGB2GRAY), np.ones((3, 3), np.uint8))
    mask = (seam_safe > strict).astype(np.uint8)
    # Text and drawings punch holes in the page: fill every large outer shape solid.
    outlines, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(mask)
    big = [c for c in outlines if cv2.contourArea(c) > 0.03 * mask.size]
    cv2.drawContours(mask, big, -1, 1, thickness=cv2.FILLED)
    erode =max(3, round(max(gray.shape) * 0.006))
    kernel = np.ones((2 * erode + 1, 2 * erode + 1), np.uint8)
    core = cv2.erode(mask, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    h, w = gray.shape
    candidates = []
    for i in range(1, count):
        x, y, cw, ch, area = stats[i]
        if area < 0.12 * h * w:
            continue
        touches = x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1
        candidates.append((touches, -area, i))
    if not candidates:
        return None
    _, _, best = min(candidates)
    blob = cv2.dilate((labels == best).astype(np.uint8), kernel)
    blob = cv2.bitwise_and(blob, mask)
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    rect = cv2.minAreaRect(hull)
    box_area = rect[1][0] * rect[1][1]
    if box_area <= 0 or cv2.contourArea(hull) / box_area < 0.85:
        return None  # not rectangular enough to be a page
    if box_area > 0.92 * h * w:
        return None  # the page already fills the picture: nothing to crop
    # A real page photo is surrounded by dark; an uneven-light scan (shadowed gutter) is not.
    polygon = np.zeros_like(blob)
    cv2.fillPoly(polygon, [np.round(cv2.boxPoints(rect)).astype(np.int32)], 1)
    ring_size = max(3, round(max(gray.shape) * 0.03))
    ring = cv2.dilate(polygon, np.ones((2 * ring_size + 1, 2 * ring_size + 1), np.uint8)) - polygon
    inside_level = float(np.median(gray[polygon > 0]))
    if ring.sum() == 0 or np.mean(gray[ring > 0] < 0.5 * inside_level) < 0.5:
        return None
    # Pull the edges in slightly so the dark page border and the book edge are left out.
    (cx, cy), (rw, rh), angle = rect
    inset = 0.02
    rect = ((cx, cy), (rw * (1 - 2 * inset), rh * (1 - 2 * inset)), angle)
    return _order_points(cv2.boxPoints(rect)) / scale

_quad_cache: dict = {}

def sibling_quad(sources: list[Path]) -> np.ndarray | None:
    """Typical page position (as 0..1 fractions of the photo) seen on the given sibling photos.

    The camera and stand never move, so a page too dark to detect (a full-bleed dark picture)
    sits where the neighbouring pages on the same side of the book were found.
    """
    found = []
    for source in sources:
        try:
            key = (str(source), source.stat().st_mtime_ns)
        except OSError:
            continue
        if key not in _quad_cache:
            if len(_quad_cache) > 4096:
                _quad_cache.clear()
            with Image.open(source) as original:
                original.draft("RGB", (1400, 1400))   # fast partial JPEG decode, enough for detection
                photo = ImageOps.exif_transpose(original).convert("RGB")
            quad = detect_page_quad(photo)
            _quad_cache[key] = None if quad is None else quad / np.array(photo.size, np.float32)
        if _quad_cache[key] is not None:
            found.append(_quad_cache[key])
    return np.median(np.stack(found), axis=0) if found else None

def crop_to_page(image: Image.Image, fallback=None) -> Image.Image:
    """Cut out the page found by detect_page_quad and straighten it.

    fallback: optional callable returning a quad (0..1 fractions) used when detection fails.
    """
    quad = detect_page_quad(image)
    if quad is None and fallback is not None:
        relative = fallback()
        if relative is not None:
            quad = (relative * np.array(image.size, np.float32)).astype(np.float32)
    if quad is None:
        return image
    tl, tr, br, bl = quad
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    if width < 50 or height < 50:
        return image
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(quad, target)
    warped = cv2.warpPerspective(np.array(image), matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return trim_neighbour_edges(Image.fromarray(warped))

MAX_EDGE_TRIM = 25.0       # most a side can be trimmed, in percent of the page size

def trim_sides(image: Image.Image, left: float = 0, top: float = 0, right: float = 0, bottom: float = 0) -> Image.Image:
    """Cut a percentage off each side of the (already upright) page: strips of the neighbouring
    page, a shadow, a reflection... that the automatic crop could not tell from the page."""
    if not any((left, top, right, bottom)):
        return image
    clamp = lambda v: min(max(float(v or 0), 0.0), MAX_EDGE_TRIM) / 100.0
    l, t = round(image.width * clamp(left)), round(image.height * clamp(top))
    r, b = image.width - round(image.width * clamp(right)), image.height - round(image.height * clamp(bottom))
    if r - l < 50 or b - t < 50:
        return image                      # never trim the page away
    return image.crop((l, t, r, b))

EDGE_DEPTH = 0.08          # how far from the border (share of the page size) a neighbour's edge is looked for
EDGE_STEP = 35             # tone jump (0-255) that makes a boundary
EDGE_VARIATION = 18        # a neighbour's edge (coloured tabs, stacked pages) varies along the strip

def trim_neighbour_edges(image: Image.Image) -> Image.Image:
    """Cut off the edge of the neighbouring pages that shows next to a dark page.

    On a page that is a dark picture right up to its border, the quad found from the sibling
    pages can reach past the paper onto the side of the page stack (coloured index tabs, the
    edges of other leaves). That strip is brighter than the picture, separated from it by a sharp
    straight boundary, and unlike a white margin it varies along its length. Only then is it cut.
    """
    scale = min(1.0, 800 / max(image.size))
    small = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.BILINEAR) if scale < 1 else image
    gray = cv2.cvtColor(np.array(small), cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape

    def cut_from_far_edge(g: np.ndarray) -> int:
        """Number of pixels to remove from the far (right) edge of g, 0 when there is no strip."""
        rows = slice(int(g.shape[0] * 0.1), int(g.shape[0] * 0.9))
        depth = int(g.shape[1] * EDGE_DEPTH)
        if depth < 8:
            return 0
        profile = g[rows, g.shape[1] - depth:].mean(axis=0)            # columns, inside -> outside
        jumps = [float(profile[x:x + 3].mean() - profile[x - 4:x].mean()) for x in range(4, depth - 2)]
        # Case 1: a dark page with a brighter, uneven strip beyond it (index tabs, stacked leaves).
        best = int(np.argmax(jumps)) + 4
        if jumps[best - 4] >= EDGE_STEP:
            start = g.shape[1] - depth + best
            inside = g[rows, max(0, start - int(g.shape[1] * 0.03)):start].mean()
            strip = g[rows, start + 2:g.shape[1]]
            if inside <= 110 and strip.shape[1] >= 3 and float(strip.mean(axis=1).std()) >= EDGE_VARIATION:
                return g.shape[1] - start
        # Light pages are left alone: their borders (shadow, book block, reflection) differ too
        # much from book to book for a rule that never cuts real content. The manual trim of the
        # four edges (edge_trim_*) covers them.
        return 0

    cuts = {
        "right": cut_from_far_edge(gray),
        "left": cut_from_far_edge(gray[:, ::-1]),
        "bottom": cut_from_far_edge(gray.T),
        "top": cut_from_far_edge(gray.T[:, ::-1]),
    }
    if not any(cuts.values()):
        return image
    left, right = round(cuts["left"] / scale), round(cuts["right"] / scale)
    top, bottom = round(cuts["top"] / scale), round(cuts["bottom"] / scale)
    return image.crop((left, top, image.width - right, image.height - bottom))

COLOR_PAGE_TARGET = 170    # brightness (0-255) a flat coloured page is brought to
COLOR_PAGE_SHARE = 0.60    # share of the page that must be one flat colour for it to count
COLOR_PAGE_SATURATION = 0.60   # vivid colours only: a dull brown or grey picture is left as it is
COLOR_PAGE_MIN_LUMINANCE = 50  # a nearly black page is a dark picture, not a coloured page that was underexposed

def lift_colored_page(image: Image.Image) -> Image.Image:
    """Brighten a page whose background is one flat, saturated colour (a yellow or green opener page).

    Such a page has no white paper to measure, so it stays as dark as the camera exposed it,
    while the white pages of the same book are lifted to white. The colour is brought to a
    mid-bright tone (at most 2.2x, hue kept) so its text becomes readable again. A page that
    is not mostly one saturated colour comes back untouched.
    """
    scale = 400 / max(image.size)
    small = np.array(image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.BILINEAR)) if scale < 1 else np.array(image)
    pixels = small.reshape(-1, 3).astype(np.float32)
    median = np.median(pixels, axis=0)
    hsv = cv2.cvtColor(np.uint8([[median]]), cv2.COLOR_RGB2HSV)[0, 0]
    flat = float(np.mean(np.linalg.norm(pixels - median, axis=1) < 45))
    if flat < COLOR_PAGE_SHARE or hsv[1] / 255.0 < COLOR_PAGE_SATURATION:
        return image
    luminance = float(median @ np.array([0.299, 0.587, 0.114], np.float32))
    if luminance < COLOR_PAGE_MIN_LUMINANCE:
        return image
    gain = min(COLOR_PAGE_TARGET / luminance, MAX_PAPER_GAIN)
    if gain < 1.1:
        return image                                    # already bright enough
    return Image.fromarray(np.clip(np.array(image).astype(np.float32) * gain, 0, 255).astype(np.uint8))

def apply_clahe(image: Image.Image) -> Image.Image:
    """Moderate local contrast (CLAHE) on the lightness channel only, so colors stay natural."""
    lab = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    equalized = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8)).apply(l)
    # CLAHE would lift solid black areas (bars, plates) to grey: keep the original there.
    paper = _paper_level(l)
    keep =np.clip((0.45 - l.astype(np.float32) / paper) / 0.2, 0, 1)
    l = np.clip(keep * l + (1 - keep) * equalized, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB))

def reduce_show_through(image: Image.Image, strength: float) -> Image.Image:
    """Fade the text of the other side (verso) that shines through thin paper.

    Show-through is much lighter than real ink, so a levels curve on the lightness sends
    everything above a cutoff to white and keeps the ink dark. Colors keep their hue.
    """
    strength = min(max(strength, 0.0), 1.0)
    rgb = np.array(image).astype(np.float32)
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    paper = _paper_level(lum)
    x = lum / paper
    low, high = 0.2 * strength, 1.0 - 0.45 * strength
    out = np.clip((x - low) / (high - low), 0, 1)
    gain = np.where(lum > 1, out * paper / np.maximum(lum, 1), 0)
    return Image.fromarray(np.clip(rgb * gain[..., None], 0, 255).astype(np.uint8))

def denoise(image: Image.Image, mode: str) -> Image.Image:
    """'median': light 3x3 median. 'edge': edge-preserving filter that keeps letter outlines."""
    rgb = np.array(image)
    if mode == "median":
        return Image.fromarray(cv2.medianBlur(rgb, 3))
    smoothed = cv2.bilateralFilter(rgb, BILATERAL_DIAMETER, 30, 30)
    if mode == "combined":
        # Edge-preserving smoothing first, then a light non-local-means pass for the grain.
        smoothed = cv2.fastNlMeansDenoisingColored(smoothed, None, NLM_STRENGTH, NLM_STRENGTH, 7, 21)
    return Image.fromarray(smoothed)

def unsharp(image: Image.Image, amount: float) -> Image.Image:
    """Unsharp mask; the threshold leaves flat paper untouched, which limits halos and noise boost.

    The blur radius is UNSHARP_SIGMA pixels on a page about 2400 px long and grows with larger pages,
    so the sharpening looks the same at any resolution (but never goes below 0.5 px, where it does nothing).
    """
    radius = max(0.5, UNSHARP_SIGMA * max(image.size) / 2400)
    return image.filter(ImageFilter.UnsharpMask(radius=radius, percent=int(amount * 100), threshold=3))

def _skew_score(binary: np.ndarray, angle: float) -> float:
    h, w = binary.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(binary, matrix, (w, h), flags=cv2.INTER_NEAREST)
    return float(np.var(rotated.sum(axis=1, dtype=np.float64)))

def estimate_skew(image: Image.Image, max_angle: float = 5.0) -> float:
    """Skew angle in degrees (projection-profile method): text lines are sharpest when level."""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    scale = 700 / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1].astype(np.float32)
    coarse = np.arange(-max_angle, max_angle + 0.01, 0.5)
    best = max(coarse, key=lambda a: _skew_score(binary, a))
    best = max(np.arange(best - 0.5, best + 0.51, 0.1), key=lambda a: _skew_score(binary, a))
    # Pages with little text give a flat score curve: leave them alone.
    if _skew_score(binary, best) < _skew_score(binary, 0.0) * 1.05:
        return 0.0
    return float(best)

def deskew(image: Image.Image) -> Image.Image:
    angle = estimate_skew(image)
    if abs(angle) < 0.15:
        return image
    rgb = np.array(image)
    h, w = rgb.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(rgb, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(rotated)

def binarize(image: Image.Image) -> Image.Image:
    """Sauvola thresholding: robust on uneven backgrounds; best for text-only pages."""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)  # suppress isolated noise specks
    window = _odd(max(gray.shape[:2]) // 40, 25)
    mean = cv2.boxFilter(gray, -1, (window, window), borderType=cv2.BORDER_REPLICATE)
    sq_mean = cv2.boxFilter(gray * gray, -1, (window, window), borderType=cv2.BORDER_REPLICATE)
    std = np.sqrt(np.maximum(sq_mean - mean * mean, 0))
    k, r = 0.2, 128.0
    threshold = mean * (1 + k * (std / r - 1))
    # Inside a solid black area the local mean is as dark as the noise, so Sauvola alone
    # would speckle it with white: anything clearly darker than the paper is simply black.
    paper = float(np.percentile(gray, 90))
    bw = np.where((gray > threshold) & (gray > 0.4 * paper), 255, 0).astype(np.uint8)
    return Image.fromarray(bw).convert("RGB")

A4_PX = (2480, 3508)  # 210 x 297 mm at 300 dpi
A4_PX_PER_MM = 300 / 25.4

def fit_to_a4(image: Image.Image, scale: float = 1.0, margin_mm: float = 5.0, align: str = "center") -> Image.Image:
    """Place the page on a white A4 sheet (portrait or landscape), keeping its proportions.

    scale < 1 produces a smaller sheet with the same layout, used for previews.
    margin_mm: white border kept around the page. align: "center" or "top" (the free space,
    when the page is not as tall as the sheet, then goes below instead of above and below).
    """
    width, height = A4_PX if image.height >= image.width else A4_PX[::-1]
    sheet_w, sheet_h = max(1, round(width * scale)), max(1, round(height * scale))
    margin = round(margin_mm * A4_PX_PER_MM * scale)
    ratio = min((sheet_w - 2 * margin) / image.width, (sheet_h - 2 * margin) / image.height)
    size = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
    resized = image.resize(size, Image.LANCZOS)
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    top = margin if align == "top" else (sheet_h - size[1]) // 2
    sheet.paste(resized, ((sheet_w - size[0]) // 2, top))
    return sheet

_rotation_cache: dict = {}

def page_rotation(source: Path, image: Image.Image, book) -> int:
    """Automatic rotation of a page, remembered per file so previews stay fast."""
    try:
        key = (str(source), source.stat().st_mtime_ns, bool(book.auto_crop_page), book.language)
    except OSError:
        return detect_rotation(image, book.language)
    if key not in _rotation_cache:
        if len(_rotation_cache) > 4096:
            _rotation_cache.clear()
        _rotation_cache[key] = detect_rotation(image, book.language)
    return _rotation_cache[key]

PHOTO_FIELDS = ("output_mode", "brightness", "contrast", "clahe", "background_clean",
                "show_through", "denoise_mode", "unsharp")

class _PhotoSettings:
    """The book's settings, with the rendering options replaced by its photo_* ones."""
    def __init__(self, book):
        self._book = book

    def __getattr__(self, name):
        if name in PHOTO_FIELDS:
            return getattr(self._book, "photo_" + name)
        return getattr(self._book, name)

def same_side_neighbours(pages, page, limit: int = 4) -> list[Path]:
    """Photos of the nearest pages on the same side of the book (same parity of order)."""
    others = [p for p in pages if p.pk != page.pk and (p.order - page.order) % 2 == 0]
    others.sort(key=lambda p: abs(p.order - page.order))
    return [Path(p.image.path) for p in others[:limit]]

def process_image(source: Path, book, max_side: int | None = None, extra_rotation: int = 0,
                  fallback_sources: list[Path] | None = None, photo_page: bool = False,
                  page_trim: tuple | None = None) -> Image.Image:
    """Apply the book's settings. With max_side, work on a downscaled copy (preview).

    page_trim: (left, top, right, bottom) in percent for this page only, replacing the book's trim.

    extra_rotation: manual clockwise turn (multiple of 90) chosen for this page by the user.
    fallback_sources: sibling photos used to locate the page when it cannot be detected itself.
    photo_page: the page has a photo, so the book's photo_* rendering settings apply.
    """
    if photo_page:
        book = _PhotoSettings(book)
    with Image.open(source) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
    if max_side and max(image.size) > 3 * max_side:
        image.thumbnail((3 * max_side, 3 * max_side), Image.LANCZOS)  # keeps preview fast on huge photos
    if book.auto_crop_page:
        # Done before the final downscale so the cut-out page keeps its detail.
        fallback = (lambda: sibling_quad(fallback_sources)) if fallback_sources else None
        image = crop_to_page(image, fallback)
    turn = page_rotation(source, image, book) if book.auto_rotate else 0
    image = rotate_clockwise(image, turn + extra_rotation)
    left, top, right, bottom = page_trim or (book.edge_trim_left, book.edge_trim_top, book.edge_trim_right, book.edge_trim_bottom)
    image = trim_sides(image, left, top, right, bottom)
    base_side = max(image.size)
    if max_side and base_side > max_side:
        image.thumbnail((max_side, max_side), Image.LANCZOS)
    if book.lift_colored_pages:
        image = lift_colored_page(image)
    if book.brightness != 1.0:
        image = ImageEnhance.Brightness(image).enhance(book.brightness)
    if book.contrast != 1.0:
        image = ImageEnhance.Contrast(image).enhance(book.contrast)
    if book.clahe:
        image = apply_clahe(image)
    if book.background_clean:
        image = clean_background(image)
    if book.denoise_mode != "off":
        image = denoise(image, book.denoise_mode)
    if book.show_through > 0:
        image = reduce_show_through(image, book.show_through)
    if book.deskew:
        image = deskew(image)
    if book.unsharp > 0:
        image = unsharp(image, book.unsharp)
    if book.output_mode == "gray":
        image = ImageOps.grayscale(image).convert("RGB")
    elif book.output_mode == "bw":
        image = binarize(image)
    if book.crop_margins:
        padding = book.crop_padding
        if max_side:
            # Keep the padding proportional on the downscaled preview.
            padding = round(padding * max(image.size) / base_side)
        image = crop_page(image, padding)
    if book.page_format == "a4":
        image = fit_to_a4(image, scale=max_side / A4_PX[1] if max_side else 1.0,
                          margin_mm=book.a4_margin_mm, align=book.a4_align)
    return image

def preprocess_image(source: Path, destination: Path, book, extra_rotation: int = 0,
                     fallback_sources: list[Path] | None = None, photo_page: bool = False,
                     page_trim: tuple | None = None) -> None:
    process_image(source, book, extra_rotation=extra_rotation, fallback_sources=fallback_sources,
                  photo_page=photo_page, page_trim=page_trim).save(destination, "PNG", dpi=(300, 300), optimize=True)
