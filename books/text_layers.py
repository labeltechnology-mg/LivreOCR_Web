"""PaddleOCR-guided text and background layers for the page preview."""
from threading import Lock

import cv2
import numpy as np
from PIL import Image


MODEL_NAME = "PP-OCRv6_medium_det"
_detector = None
_detector_lock = Lock()
_inference_lock = Lock()


class TextLayerError(Exception):
    status_code = 503


class TextNotDetected(TextLayerError):
    status_code = 422


def _load_detector():
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                try:
                    from paddleocr import TextDetection
                except ImportError as exc:
                    raise TextLayerError(
                        "La séparation des couches nécessite PaddleOCR. "
                        "Installez les dépendances PaddleOCR indiquées dans le README."
                    ) from exc
                try:
                    _detector = TextDetection(
                        model_name=MODEL_NAME,
                        device="cpu",
                        limit_side_len=2000,
                        limit_type="max",
                    )
                except Exception as exc:
                    raise TextLayerError(
                        "Impossible de charger le détecteur PaddleOCR. Vérifiez la connexion "
                        f"au premier téléchargement du modèle. Détail : {str(exc)[:300]}"
                    ) from exc
    return _detector


def _prediction(detector, rgb: np.ndarray) -> tuple[list, list]:
    try:
        with _inference_lock:
            predictions = detector.predict(rgb)
            prediction = next(iter(predictions), None)
            payload = prediction.json if prediction is not None else {}
        result = payload.get("res", payload)
        return result.get("dt_polys", []), result.get("dt_scores", [])
    except Exception as exc:
        raise TextLayerError(f"Échec de la détection PaddleOCR : {str(exc)[:300]}") from exc


def _horizontal_ratio(polygons, scores) -> float:
    eligible = [
        (np.asarray(polygon), float(scores[index]) if index < len(scores) else 1.0)
        for index, polygon in enumerate(polygons)
        if np.asarray(polygon).size >= 8
    ]
    if not eligible:
        return 0.0
    horizontal = 0.0
    total = 0.0
    for polygon, score in eligible:
        x_span = float(np.ptp(polygon[:, 0]))
        y_span = float(np.ptp(polygon[:, 1]))
        total += score
        if x_span >= 1.4 * y_span:
            horizontal += score
    return horizontal / total if total else 0.0


def _detect_lines(image: Image.Image, detector) -> tuple[np.ndarray, list, list, bool]:
    rgb = np.asarray(image.convert("RGB"))
    polygons, scores = _prediction(detector, rgb)
    rotated = False
    if polygons and _horizontal_ratio(polygons, scores) < 0.65:
        rotated_rgb = np.rot90(rgb, 1).copy()
        rotated_polygons, rotated_scores = _prediction(detector, rotated_rgb)
        if _horizontal_ratio(rotated_polygons, rotated_scores) > _horizontal_ratio(polygons, scores):
            rgb, polygons, scores, rotated = rotated_rgb, rotated_polygons, rotated_scores, True
    return rgb, polygons, scores, rotated


def _text_mask(rgb: np.ndarray, polygons, scores) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    block = min(75, max(15, max(height, width) // 18))
    if block % 2 == 0:
        block += 1
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    mask = np.zeros_like(gray)
    accepted = 0

    for index, polygon in enumerate(polygons):
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(points) < 4:
            continue
        confidence = float(scores[index]) if index < len(scores) else 1.0
        if confidence < 0.45:
            continue
        padding = block // 2
        x0 = max(0, int(np.floor(points[:, 0].min())) - padding)
        y0 = max(0, int(np.floor(points[:, 1].min())) - padding)
        x1 = min(width, int(np.ceil(points[:, 0].max())) + padding + 1)
        y1 = min(height, int(np.ceil(points[:, 1].max())) + padding + 1)
        crop_points = np.round(points - np.array([x0, y0])).astype(np.int32)
        line_region = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillPoly(line_region, [crop_points], 255)
        gray_crop = gray[y0:y1, x0:x1]
        smooth_crop = smooth[y0:y1, x0:x1]
        selected = line_region > 0
        if not np.any(selected):
            continue
        polarity = cv2.THRESH_BINARY_INV if np.median(gray_crop[selected]) >= 127 else cv2.THRESH_BINARY
        candidates = cv2.adaptiveThreshold(
            smooth_crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, polarity, block, 8,
        )
        candidates = cv2.bitwise_and(candidates, line_region)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
        crop_mask = mask[y0:y1, x0:x1]
        for label in range(1, component_count):
            _, _, _, component_height, area = stats[label]
            if area >= 2 and component_height <= max(60, height * 0.06):
                crop_mask[labels == label] = 255
        accepted += 1

    if not accepted or not np.any(mask):
        raise TextNotDetected("PaddleOCR n’a détecté aucun caractère exploitable sur cette page.")
    return mask


def separate_text_background(image: Image.Image, detector=None) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Return text-only, text-removed background, and overlay preview images."""
    original_size = image.size
    source = image.convert("RGB")
    scale = min(2.0, 1000 / max(source.size)) if max(source.size) < 1000 else 1.0
    if scale > 1:
        source = source.resize(
            (round(source.width * scale), round(source.height * scale)),
            Image.Resampling.LANCZOS,
        )

    rgb, polygons, scores, rotated = _detect_lines(source, detector or _load_detector())
    if not polygons:
        raise TextNotDetected("PaddleOCR n’a détecté aucune ligne de texte sur cette page.")
    text_mask = _text_mask(rgb, polygons, scores)
    expanded_mask = cv2.dilate(text_mask, np.ones((3, 3), np.uint8))
    background = cv2.inpaint(rgb, expanded_mask, 4, cv2.INPAINT_TELEA)
    text = np.full_like(rgb, 255)
    text[text_mask > 0] = 0
    overlay = background.copy()
    overlay[text_mask > 0] = (0.45 * overlay[text_mask > 0] + np.array([220, 35, 45]) * 0.55).astype(np.uint8)

    if rotated:
        text, background, overlay = (np.rot90(layer, 3).copy() for layer in (text, background, overlay))
    if scale > 1:
        target = original_size
        text = cv2.resize(text, target, interpolation=cv2.INTER_AREA)
        background = cv2.resize(background, target, interpolation=cv2.INTER_AREA)
        overlay = cv2.resize(overlay, target, interpolation=cv2.INTER_AREA)
    return Image.fromarray(text), Image.fromarray(background), Image.fromarray(overlay)
