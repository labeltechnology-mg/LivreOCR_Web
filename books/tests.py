import io
import json
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from PIL import Image, ImageDraw

from .image_processing import (
    A4_PX, adjust_brightness, apply_page_layout, detect_page_quad, estimate_skew, fit_to_a4,
    prepare_page_image, process_image, reduce_show_through,
)
from .models import Book, PageImage, PriceSettings
from .orientation import _run as run_tesseract, find_tesseract
from .text_layers import TextLayerError, TextNotDetected, _text_mask, separate_text_background

TEXT = "Texte de test pour la detection des pages dans un livre"


def text_page(width=1200, height=1600, fill="white", skew=0.0) -> Image.Image:
    image = Image.new("RGB", (width, height), fill)
    draw = ImageDraw.Draw(image)
    for row in range(30):
        draw.text((120, 150 + row * 40), TEXT, fill="black")
    return image.rotate(skew, fillcolor=fill) if skew else image


def photo_on_dark_stand() -> Image.Image:
    """A bright page lying on a dark plate, like the scanning stand."""
    photo = Image.new("RGB", (2000, 1333), (20, 20, 20))
    photo.paste(text_page(900, 620, (160, 160, 160)), (550, 300))
    return photo


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


class FakeTextDetector:
    def __init__(self, polygons=True):
        self.polygons = polygons

    def predict(self, image):
        height, width = image.shape[:2]
        polygons = []
        scores = []
        if self.polygons:
            polygons = [np.array([
                [0.1 * width, 0.18 * height],
                [0.6 * width, 0.18 * height],
                [0.6 * width, 0.32 * height],
                [0.1 * width, 0.32 * height],
            ], dtype=np.float32)]
            scores = [0.9]
        return [SimpleNamespace(json={"res": {"dt_polys": polygons, "dt_scores": scores}})]


class ImageProcessingTests(TestCase):
    def test_brightness_adjustment_preserves_highlight_details(self):
        gradient = np.tile(np.arange(256, dtype=np.uint8), (8, 1))
        page = Image.fromarray(np.repeat(gradient[:, :, None], 3, axis=2))

        brighter = np.array(adjust_brightness(page, 1.3).convert("L"))
        darker = np.array(adjust_brightness(page, 0.7).convert("L"))

        self.assertTrue(np.array_equal(np.array(adjust_brightness(page, 1.0)), np.array(page)))
        self.assertGreater(brighter[0, 180], 180)
        self.assertLess(brighter[0, 240], 255)
        self.assertEqual(brighter[0, 255], 255)
        self.assertLess(darker[0, 180], 180)
        self.assertTrue(np.all(np.diff(brighter[0]) >= 0))

    @patch("books.orientation.subprocess.run")
    def test_tesseract_output_uses_utf8_without_platform_decode_errors(self, run):
        run.return_value = SimpleNamespace(returncode=0, stdout="Texte français")

        output = run_tesseract("tesseract", Image.new("RGB", (10, 10)), [])

        self.assertEqual(output, "Texte français")
        self.assertEqual(run.call_args.kwargs["encoding"], "utf-8")
        self.assertEqual(run.call_args.kwargs["errors"], "replace")

    def test_text_and_background_layers_are_extracted(self):
        page = Image.new("RGB", (600, 800), (245, 240, 225))
        ImageDraw.Draw(page).text((80, 180), "Printed text", fill=(20, 20, 20))

        text, background, overlay = separate_text_background(page, FakeTextDetector())

        original_pixels = np.array(page)
        text_pixels = np.array(text.convert("L"))
        background_pixels = np.array(background)
        self.assertGreater(np.count_nonzero(text_pixels[160:220, 60:320] < 128), 10)
        self.assertTrue(np.all(text_pixels[:150] == 255))  # extracted text is shown alone on white
        self.assertGreater(np.count_nonzero(
            np.any(original_pixels[160:220, 60:320] != background_pixels[160:220, 60:320], axis=2)
        ), 10)  # the detected text is removed from the background layer
        self.assertTrue(np.array_equal(background_pixels[30, 30], original_pixels[30, 30]))
        self.assertFalse(np.array_equal(np.array(overlay), np.array(background)))
        self.assertEqual(text.size, page.size)
        self.assertEqual(background.size, page.size)

    def test_light_text_is_extracted_from_a_dark_background(self):
        page = Image.new("RGB", (600, 800), (40, 35, 30))
        ImageDraw.Draw(page).text((80, 180), "Light text", fill=(240, 240, 235))

        text, background, _ = separate_text_background(page, FakeTextDetector())

        text_pixels = np.array(text.convert("L"))
        original_pixels = np.array(page)
        background_pixels = np.array(background)
        self.assertGreater(np.count_nonzero(text_pixels[160:220, 60:320] < 128), 10)
        self.assertTrue(np.all(text_pixels[:150] == 255))
        self.assertGreater(np.count_nonzero(
            np.any(original_pixels[160:220, 60:320] != background_pixels[160:220, 60:320], axis=2)
        ), 10)

    def test_text_mask_keeps_wide_connected_glyphs_inside_detected_line(self):
        page = np.full((400, 600, 3), 255, dtype=np.uint8)
        cv2.rectangle(page, (100, 190), (240, 210), (0, 0, 0), thickness=-1)
        polygon = np.array([[80, 175], [280, 175], [280, 225], [80, 225]], dtype=np.float32)

        mask = _text_mask(page, [polygon], [0.95])

        self.assertGreater(np.count_nonzero(mask[185:215, 90:250]), 2000)

    def test_text_layers_report_when_no_text_is_detected(self):
        page = Image.new("RGB", (100, 100), "white")
        with self.assertRaises(TextNotDetected):
            separate_text_background(page, FakeTextDetector(polygons=False))

    def test_a4_sheet_follows_page_orientation(self):
        self.assertEqual(fit_to_a4(Image.new("RGB", (900, 1400))).size, A4_PX)
        self.assertEqual(fit_to_a4(Image.new("RGB", (1400, 900))).size, A4_PX[::-1])

    def test_page_preparation_and_final_layout_are_distinct_stages(self):
        from .models import Book

        page = text_page(300, 400)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "page.png"
            page.save(path)
            settings = Book(auto_crop_page=False, auto_rotate=False, deskew=False,
                            crop_margins=False, page_format="a4", output_mode="color")

            adjusted, base_side = prepare_page_image(path, settings, max_side=600)
            final = apply_page_layout(adjusted.copy(), settings, max_side=600, base_side=base_side)
            full = process_image(path, settings, max_side=600)

        self.assertEqual(adjusted.size, page.size)
        self.assertEqual(final.size, (round(A4_PX[0] * 600 / A4_PX[1]), 600))
        self.assertEqual(final.size, full.size)
        self.assertNotEqual(adjusted.size, final.size)

    def test_a4_margin_and_alignment(self):
        wide = Image.new("RGB", (1000, 1100), "black")           # squarer than A4: free space above and below
        sheet = np.array(fit_to_a4(wide, margin_mm=10, align="top").convert("L"))
        rows = np.where(sheet.min(axis=1) == 0)[0]
        self.assertAlmostEqual(rows[0], round(10 * 300 / 25.4), delta=2)          # starts at the margin
        self.assertGreater(sheet.shape[0] - rows[-1], 800)                        # free space is all below
        centred = np.array(fit_to_a4(wide, margin_mm=10).convert("L"))
        rows = np.where(centred.min(axis=1) == 0)[0]
        self.assertAlmostEqual(rows[0], centred.shape[0] - rows[-1], delta=3)     # balanced above and below

    def test_quality_chain_runs_and_keeps_text_dark_and_paper_light(self):
        from . import image_processing as ip
        self.assertEqual((ip.CLAHE_CLIP_LIMIT, ip.BILATERAL_DIAMETER, ip.NLM_STRENGTH, ip.UNSHARP_SIGMA), (1.45, 5, 2.5, 0.65))
        page = text_page(600, 800)
        rng = np.random.default_rng(3)
        noisy = Image.fromarray((np.array(page).astype(float) + rng.normal(0, 6, (800, 600, 3))).clip(0, 255).astype("uint8"))
        smoothed = ip.denoise(noisy, "combined")
        self.assertLess(np.array(smoothed.convert("L"))[:100].std(), np.array(noisy.convert("L"))[:100].std())   # grain reduced
        result = ip.unsharp(ip.apply_clahe(smoothed), 1.22)
        gray = np.array(result.convert("L"))
        self.assertEqual(result.size, page.size)
        self.assertLess(gray.min(), 60)                    # ink still dark
        self.assertGreater(np.median(gray), 200)           # paper still light

    def test_flat_coloured_pages_are_brightened_and_nothing_else(self):
        from .image_processing import lift_colored_page
        rng = np.random.default_rng(5)

        def flat(colour, size=(400, 500)):
            noise = rng.normal(0, 4, (size[1], size[0], 3))
            return Image.fromarray((np.array(colour, float) + noise).clip(0, 255).astype("uint8"))

        olive = flat((103, 113, 4))                                        # dark yellow-green page
        lifted = np.array(lift_colored_page(olive)).reshape(-1, 3).mean(axis=0)
        self.assertGreater(lifted[1], 170)                                 # clearly brighter
        self.assertGreater(lifted[1], lifted[0])                           # still yellow-green: hue kept
        self.assertLess(lifted[2], 20)
        for untouched in (flat((255, 255, 255)), flat((200, 195, 190)),    # white and dull paper
                          flat((31, 24, 17)),                              # nearly black picture
                          flat((235, 220, 20))):                           # already bright yellow
            self.assertTrue(np.array_equal(np.array(lift_colored_page(untouched)), np.array(untouched)))
        self.assertTrue(np.array_equal(np.array(lift_colored_page(text_page(300, 400))), np.array(text_page(300, 400))))   # a text page

    def test_yellow_background_is_neutralized_without_erasing_ink_or_artwork(self):
        from .image_processing import neutralize_yellow_background

        page = Image.new("RGB", (500, 600), (242, 223, 145))
        draw = ImageDraw.Draw(page)
        draw.text((80, 150), "Texte de test", fill=(20, 15, 10))
        draw.rectangle((350, 350, 450, 450), fill=(255, 170, 0))
        draw.rectangle((350, 470, 450, 550), fill=(255, 230, 0))
        result = np.array(neutralize_yellow_background(page))

        background = result[30:100, 30:100].mean(axis=0)
        self.assertGreater(background.mean(), 245)
        self.assertLess(background.max() - background.min(), 2)
        text_region = result[145:165, 75:220]
        self.assertGreater((text_region.mean(axis=2) < 100).sum(), 20)
        self.assertTrue(np.array_equal(result[400, 400], np.array([255, 170, 0])))
        self.assertTrue(np.array_equal(result[500, 400], np.array([255, 230, 0])))
        self.assertTrue(np.array_equal(
            np.array(neutralize_yellow_background(Image.new("RGB", (100, 100), (200, 195, 190)))),
            np.full((100, 100, 3), (200, 195, 190), dtype=np.uint8),
        ))

    def test_yellow_cleanup_preserves_dark_brown_ink(self):
        from .image_processing import neutralize_yellow_background

        page = Image.new("RGB", (500, 500), (244, 230, 175))
        draw = ImageDraw.Draw(page)
        draw.text((80, 200), "Francoise Bonvin", fill=(95, 73, 49))
        before = np.array(page)
        after = np.array(neutralize_yellow_background(page))
        ink = before[200:215, 80:230, 0] < 140
        self.assertGreater(ink.sum(), 10)
        self.assertLess(float(after[200:215, 80:230, 0][ink].mean()), 140)

    def test_yellow_cleanup_preserves_brown_artwork(self):
        from .image_processing import neutralize_yellow_background

        page = Image.new("RGB", (600, 500), (244, 230, 175))
        draw = ImageDraw.Draw(page)
        draw.rectangle((80, 100, 260, 400), fill=(190, 150, 110))
        draw.rectangle((340, 100, 520, 400), fill=(170, 140, 110))

        result = np.array(neutralize_yellow_background(page))

        self.assertTrue(np.array_equal(result[250, 170], np.array((190, 150, 110))))
        self.assertTrue(np.array_equal(result[250, 430], np.array((170, 140, 110))))
        self.assertGreater(result[20, 20].mean(), 245)

    def test_stained_yellow_paper_is_whitened_without_losing_text(self):
        from .image_processing import neutralize_yellow_background

        page = Image.new("RGB", (500, 600), (244, 230, 175))
        draw = ImageDraw.Draw(page)
        rng = np.random.default_rng(14)
        for _ in range(1200):
            x, y = int(rng.integers(0, 500)), int(rng.integers(0, 600))
            draw.ellipse((x, y, x + 2, y + 2), fill=(211, 194, 135))
        draw.text((80, 200), "Bonvin de Lens", fill=(55, 43, 30))

        result = np.array(neutralize_yellow_background(page))
        stain_region = result[20:180, 20:480]
        self.assertGreater(float(np.percentile(stain_region.mean(axis=2), 10)), 245)
        text_region = result[195:210, 75:200]
        self.assertGreater((text_region.mean(axis=2) < 120).sum(), 15)

    def test_yellow_background_setting_is_used_in_image_pipeline(self):
        from .image_processing import process_image

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "yellow.png"
            page = Image.new("RGB", (400, 500), (242, 223, 145))
            ImageDraw.Draw(page).text((50, 100), "Texte", fill="black")
            page.save(source)
            book = Book(auto_rotate=False, auto_crop_page=False, crop_margins=False, deskew=False,
                        neutralize_yellow_background=True, page_format="original")
            result = np.array(process_image(source, book))
            self.assertGreater(result[20, 20].mean(), 245)

    def test_yellow_background_pipeline_keeps_color_and_text(self):
        from .image_processing import process_image

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "stained-yellow.png"
            page = Image.new("RGB", (500, 600), (244, 230, 175))
            draw = ImageDraw.Draw(page)
            rng = np.random.default_rng(20)
            for _ in range(1000):
                x, y = int(rng.integers(0, 500)), int(rng.integers(0, 600))
                draw.ellipse((x, y, x + 2, y + 2), fill=(211, 194, 135))
            draw.text((80, 200), "Bonvin de Lens", fill=(55, 43, 30))
            draw.rectangle((200, 350, 300, 450), fill=(170, 140, 110))
            draw.rectangle((350, 350, 450, 450), fill=(255, 140, 20))
            page.save(source)
            book = Book(auto_rotate=False, auto_crop_page=False, crop_margins=False, deskew=False,
                        neutralize_yellow_background=True, output_mode="color", show_through=0.3,
                        page_format="original")

            result = np.array(process_image(source, book))
            self.assertGreater(len(np.unique(result.reshape(-1, 3), axis=0)), 2)
            text_region = result[200:215, 80:230]
            self.assertGreater(np.mean(np.max(text_region, axis=2) < 150), 0.01)
            brown = result[400, 250]
            self.assertGreater(brown[0], brown[1])
            self.assertGreater(brown[1], brown[2])
            self.assertLess(brown.mean(), 200)
            self.assertGreater(result[400, 400, 0], result[400, 400, 2])

    def test_manual_edge_trim(self):
        from .image_processing import trim_sides
        page = Image.new("RGB", (1000, 2000), "white")
        self.assertEqual(trim_sides(page).size, (1000, 2000))                              # nothing asked, nothing cut
        self.assertEqual(trim_sides(page, left=2, top=1, right=7, bottom=10).size, (910, 1780))
        self.assertEqual(trim_sides(page, left=90, right=90).size, (500, 2000))            # at most 25% per side
        self.assertEqual(trim_sides(Image.new("RGB", (80, 80)), left=25, right=25).size, (80, 80))   # never trims the page away
        self.assertEqual(trim_sides(page, left=-5).size, (1000, 2000))                     # negative values are ignored

    def test_neighbour_edge_is_trimmed_only_next_to_a_dark_page(self):
        from .image_processing import trim_neighbour_edges
        page = Image.new("RGB", (800, 1000), (35, 25, 15))
        draw = ImageDraw.Draw(page)
        for i, colour in enumerate([(70, 100, 60), (200, 150, 30), (170, 100, 60), (110, 80, 60), (235, 235, 235)] * 2):
            draw.rectangle((760, i * 100, 800, i * 100 + 99), fill=colour)     # coloured index tabs of the page stack
        trimmed = trim_neighbour_edges(page)
        self.assertLessEqual(trimmed.width, 765)
        self.assertEqual(trimmed.height, 1000)
        margin = Image.new("RGB", (800, 1000), (35, 25, 15))
        ImageDraw.Draw(margin).rectangle((760, 0, 800, 1000), fill=(245, 245, 245))   # plain white margin
        self.assertEqual(trim_neighbour_edges(margin).size, (800, 1000))
        self.assertEqual(trim_neighbour_edges(text_page()).size, (1200, 1600))         # a white text page is untouched

    def test_skew_is_detected_and_straight_pages_are_left_alone(self):
        self.assertAlmostEqual(abs(estimate_skew(text_page(skew=2.0))), 2.0, delta=0.4)
        self.assertEqual(estimate_skew(text_page()), 0.0)

    def test_page_is_found_on_dark_stand_and_not_invented_on_a_scan(self):
        quad = detect_page_quad(photo_on_dark_stand())
        self.assertIsNotNone(quad)
        width = np.linalg.norm(quad[1] - quad[0])
        self.assertAlmostEqual(width, 900, delta=90)
        self.assertIsNone(detect_page_quad(text_page()))  # a plain scan fills the frame: no crop

    def test_show_through_fades_light_marks_but_keeps_ink(self):
        page = Image.new("RGB", (200, 200), (200, 200, 200))
        ImageDraw.Draw(page).rectangle((20, 20, 60, 60), fill=(10, 10, 10))     # ink
        ImageDraw.Draw(page).rectangle((120, 120, 180, 180), fill=(170, 170, 170))  # verso ghost
        out = np.array(reduce_show_through(page, 0.8).convert("L"))
        self.assertLess(out[40, 40], 40)         # ink stays dark
        self.assertGreaterEqual(out[150, 150], 195)  # ghost goes back to the paper tone (200), was 170

    def test_process_image_gives_a4_and_photo_settings_differ(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "page.png"
            text_page(skew=0.0).save(source)
            book = Book(auto_rotate=False, auto_crop_page=False, background_clean=True, clahe=True, output_mode="bw",
                        photo_output_mode="color", photo_background_clean=False, photo_clahe=False, page_format="original")
            text = process_image(source, book, max_side=800)
            photo = process_image(source, book, max_side=800, photo_page=True)
            self.assertEqual(text.size, photo.size)
            self.assertLessEqual(len(np.unique(np.array(text.convert("L")))), 2)   # pure black and white
            self.assertGreater(len(np.unique(np.array(photo.convert("L")))), 2)    # colour/grey levels kept
            book.page_format = "a4"
            self.assertEqual(process_image(source, book, max_side=800).size[1], 800)  # long side of the A4 sheet


class ClassifyTests(TestCase):
    def test_page_types(self):
        from .classify import CHART, PHOTO, TEXT, classify_page

        self.assertEqual(classify_page(text_page())[0], TEXT)

        # A picture: smooth, textured, wide range of tones.
        rng = np.random.default_rng(1)
        y, x = np.mgrid[0:500, 0:700]
        picture = (110 + 90 * np.sin(x / 55.0) * np.cos(y / 40.0) + rng.normal(0, 14, x.shape)).clip(0, 255).astype("uint8")
        with_picture = text_page()
        with_picture.paste(Image.fromarray(picture).convert("RGB"), (250, 300))
        self.assertEqual(classify_page(with_picture)[0], PHOTO)

        # Big black bars (bar chart).
        bars = text_page(); draw = ImageDraw.Draw(bars)
        for i in range(5):
            draw.rectangle((200 + i * 160, 500 + i * 40, 320 + i * 160, 1000), fill="black")
        self.assertEqual(classify_page(bars)[0], CHART)

        # A line drawing: an axis frame and a curve, few letters.
        curve = Image.new("RGB", (1200, 1600), "white"); draw = ImageDraw.Draw(curve)
        draw.rectangle((150, 300, 1050, 1100), outline="black", width=4)
        draw.line([(150 + i * 15, 900 - int(250 * abs(np.sin(i / 7.0)))) for i in range(61)], fill="black", width=4)
        self.assertEqual(classify_page(curve)[0], CHART)

        # Text on a flat coloured box is still a text page.
        boxed = text_page(); draw = ImageDraw.Draw(boxed)
        draw.rectangle((100, 900, 1100, 1300), fill=(130, 80, 40))
        for row in range(8):
            draw.text((140, 930 + row * 40), TEXT, fill="white")
        self.assertEqual(classify_page(boxed)[0], TEXT)

    def test_grid_is_classified_as_table_not_chart(self):
        from .classify import TABLE, classify_page

        page = Image.new("RGB", (1200, 1600), "white")
        draw = ImageDraw.Draw(page)
        x0, y0, x1, y1 = 200, 400, 1000, 1000
        for y in range(y0, y1 + 1, 120):
            draw.line((x0, y, x1, y), fill="black", width=3)
        for x in range(x0, x1 + 1, 200):
            draw.line((x, y0, x, y1), fill="black", width=3)
        for row in range(4):
            for column in range(3):
                draw.text((x0 + 15 + column * 200, y0 + 25 + row * 120), f"{row + 1} {column + 1}", fill="black")

        self.assertEqual(classify_page(page)[0], TABLE)


class YieldTests(TestCase):
    """Page types, their correction, and the yield (rendement) computed from them and the prices."""

    def setUp(self):
        user_model = get_user_model()
        self.owner = user_model.objects.create_user("seller", password="pw")
        self.other = user_model.objects.create_user("stranger", password="pw")
        self.book = Book.objects.create(owner=self.owner, title="Livre", currency="Ar", price_text=100,
                                        price_table=250, price_chart=300, price_photo=400)
        media = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, media, ignore_errors=True)
        override = override_settings(MEDIA_ROOT=media)
        override.enable()
        self.addCleanup(override.disable)
        self.pages = []
        for order, kind in enumerate(["text", "text", "table", "chart", "photo", ""], 1):
            self.pages.append(PageImage.objects.create(
                book=self.book, order=order, original_name=f"p{order}.png", page_type=kind,
                image=SimpleUploadedFile(f"p{order}.png", png_bytes(text_page(100, 140)))))

    def test_yield_is_pages_times_price_by_type(self):
        from decimal import Decimal
        summary = self.book.yield_summary()
        counts = {row["type"]: row["count"] for row in summary["rows"]}
        self.assertEqual(counts, {"text": 2, "table": 1, "chart": 1, "photo": 1})
        self.assertEqual(summary["total"], Decimal(2 * 100 + 250 + 300 + 400))
        self.assertEqual(summary["analysed"], 5)
        self.assertEqual(summary["unanalysed"], 1)                       # reported apart, never priced
        self.assertEqual(summary["average"], Decimal(1150) / 5)

    def test_yield_of_a_book_without_pages_or_prices_is_zero(self):
        from decimal import Decimal
        empty = Book.objects.create(owner=self.owner, title="Vide")
        summary = empty.yield_summary()
        self.assertEqual((summary["total"], summary["average"], summary["pages"]), (Decimal(0), Decimal(0), 0))

    def test_changing_a_page_type_changes_the_yield(self):
        from decimal import Decimal
        self.client.force_login(self.owner)
        url = f"/livres/{self.book.pk}/pages/{self.pages[5].pk}/type/"       # the page that was not analysed
        self.assertEqual(self.client.post(url, {"type": "photo"}).json()["type"], "photo")
        self.assertEqual(self.book.yield_summary()["total"], Decimal(1550))
        self.assertEqual(self.book.yield_summary()["unanalysed"], 0)
        self.assertEqual(self.client.post(url, {"type": ""}).json()["type"], "")      # back to "not analysed"
        self.assertEqual(self.client.post(url, {"type": "invented"}).status_code, 400)
        self.pages[5].refresh_from_db()
        self.assertEqual(self.pages[5].page_type, "")

    def test_only_the_owner_can_change_a_page_type(self):
        self.client.force_login(self.other)
        response = self.client.post(f"/livres/{self.book.pk}/pages/{self.pages[0].pk}/type/", {"type": "photo"})
        self.assertEqual(response.status_code, 404)
        self.pages[0].refresh_from_db()
        self.assertEqual(self.pages[0].page_type, "text")

    def test_book_page_and_dashboard_show_the_yield(self):
        self.client.force_login(self.owner)
        detail = self.client.get(f"/livres/{self.book.pk}/")
        self.assertContains(detail, "Rendement total")
        self.assertContains(detail, "1150,00 Ar")                         # 5 priced pages, localised decimals
        self.assertContains(detail, "1 page(s) non analysée(s)")
        self.assertContains(self.client.get("/"), "Rendement estimé")

    def test_negative_prices_are_refused(self):
        from .forms import PriceSettingsForm, ProcessingSettingsForm
        data = {"price_text": "-1", "price_table": "1", "price_chart": "1", "price_photo": "1", "currency": "Ar"}
        self.assertFalse(PriceSettingsForm(data).is_valid())
        self.assertIn("price_text", PriceSettingsForm(data).errors)
        self.assertIn("price_text", ProcessingSettingsForm(data, instance=self.book).errors)
        data["price_text"] = "0"
        self.assertTrue(PriceSettingsForm(data).is_valid())                # zero (free) stays allowed


class TocParsingTests(TestCase):
    @staticmethod
    def line(text, y=700, x1=520):
        from .toc import Line
        return Line(text, 80, y, x1, y + 12)

    def test_entries_with_dot_leaders_are_found(self):
        from .toc import toc_entries
        lines = [self.line("Table des matières", 780), self.line("Introduction ........ 3", 700),
                 self.line("Le premier chapitre ........ 15", 660), self.line("Le second chapitre ........ 28", 620),
                 self.line("Conclusion ........ 40", 580)]
        entries, heading = toc_entries(595, lines)
        self.assertTrue(heading)
        self.assertEqual([(e.title, e.number) for e in entries],
                         [("Introduction", 3), ("Le premier chapitre", 15), ("Le second chapitre", 28), ("Conclusion", 40)])

    def test_rows_of_figures_and_plain_text_are_not_a_table_of_contents(self):
        from .toc import toc_entries
        figures = [self.line(f"{1880 + 5 * i}  1'325'000  89'000  7'050  {i + 3}", 700 - 30 * i) for i in range(8)]
        self.assertEqual(toc_entries(595, figures)[0], [])
        prose = [self.line(text, 700 - 30 * i) for i, text in enumerate([
            "Le décret fut voté en 1828", "puis appliqué jusqu'en 1838", "et repris en mai 1840", "avant la loi de 1844"])]
        self.assertEqual(toc_entries(595, prose)[0], [])

    def test_misread_numbers_going_backwards_are_dropped(self):
        from .toc import toc_entries
        lines = [self.line("Sommaire", 780)] + [self.line(f"{title} ........ {number}", 700 - 40 * i)
                 for i, (title, number) in enumerate([("Premier", 3), ("Deuxième", 5), ("Troisième", 2), ("Quatrième", 9), ("Cinquième", 12)])]
        self.assertEqual([e.number for e in toc_entries(595, lines)[0]], [3, 5, 9, 12])

    def test_page_offset_only_when_pages_are_consecutive(self):
        from .toc import _offset
        self.assertEqual(_offset({3: 2, 4: 3, 5: 4}), -1)
        self.assertIsNone(_offset({35: 1, 39: 2, 45: 3}))   # scattered pages of a partial book
        self.assertIsNone(_offset({3: 2, 4: 3}))            # too few pages to be sure

    def test_printed_numbers_forms(self):
        from .toc import _numbers_in
        self.assertEqual(_numbers_in("16 |"), [16])
        self.assertEqual(_numbers_in("| 17"), [17])
        self.assertEqual(_numbers_in("page 35"), [35])
        self.assertEqual(_numbers_in("Blickpunkt Geschichte"), [])


@unittest.skipUnless(shutil.which("ocrmypdf") and find_tesseract(), "ocrmypdf and Tesseract are required")
class TocEndToEndTests(TestCase):
    def test_table_of_contents_becomes_clickable(self):
        import pikepdf
        from PIL import ImageFont
        from .ocr import build_searchable_pdf

        def font(names, size):
            for name in names:
                try:
                    return ImageFont.truetype(name, size)
                except OSError:
                    continue
            self.skipTest("no TrueType font available to draw the test pages")

        width, height = 2480, 3508
        body_font = font(("DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"), 56)
        title_font = font(("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "LiberationSans-Bold.ttf"), 90)

        def blank():
            image = Image.new("RGB", (width, height), "white")
            return image, ImageDraw.Draw(image)

        contents, draw = blank()
        draw.text((300, 380), "Table des matières", fill="black", font=title_font)
        for i, (title, number) in enumerate([("Introduction", 3), ("Le premier chapitre", 4), ("Le second chapitre", 5)]):
            y = 720 + i * 150
            draw.text((300, y), title, fill="black", font=body_font)
            draw.text((width - 300 - draw.textlength(str(number), font=body_font), y), str(number), fill="black", font=body_font)
            x0 = 300 + draw.textlength(title, font=body_font) + 40
            x1 = width - 300 - draw.textlength(str(number), font=body_font) - 40
            draw.text((x0, y), "." * int((x1 - x0) / draw.textlength(".", font=body_font)), fill="black", font=body_font)
        images = [contents]
        for number in (2, 3, 4, 5):
            page, draw = blank()
            draw.text((300, 450), f"Chapitre page {number}", fill="black", font=title_font)
            for row in range(8):
                draw.text((300, 800 + row * 100), "Le texte courant de cette page du livre de test.", fill="black", font=body_font)
            draw.text((width // 2 - 20, height - 330), str(number), fill="black", font=body_font)
            images.append(page)

        with tempfile.TemporaryDirectory() as tmp:
            pages = []
            for order, image in enumerate(images, 1):
                path = Path(tmp) / f"s{order}.png"
                image.save(path)
                pages.append(type("P", (), {"image": type("I", (), {"path": str(path)}), "rotation": 0,
                                            "has_photo": False, "custom_trim": None, "order": order, "pk": order})())
            book = Book(language="fra", auto_rotate=False, auto_crop_page=False, crop_margins=False,
                        deskew=False, page_format="a4", toc_links=True)
            output = Path(tmp) / "out.pdf"
            stats = build_searchable_pdf(book, pages, output)
            with pikepdf.open(output) as pdf:
                targets = []
                for annotation in pdf.pages[0].get("/Annots", []):
                    destination = annotation["/Dest"][0]
                    targets.append([i for i, p in enumerate(pdf.pages) if p.objgen == destination.objgen][0] + 1)
        self.assertEqual(stats["toc_pages"], 1)
        self.assertEqual(sorted(targets), [3, 4, 5])


class OllamaRecommendationTests(TestCase):
    def recommendation(self):
        return {
            "settings": {
                "language": "deu",
                "lift_colored_pages": False,
                "neutralize_yellow_background": True,
                "output_mode": "color",
                "brightness": 1.05,
                "contrast": 1.1,
                "clahe": True,
                "background_clean": False,
                "show_through": 0.3,
                "denoise_mode": "edge",
                "unsharp": 0.6,
            },
            "reason": "Le papier est jauni, mais le texte et les encres colorées sont à préserver.",
        }

    @override_settings(OLLAMA_BASE_URL="http://127.0.0.1:11434", OLLAMA_VISION_MODEL="vision-test", OLLAMA_TIMEOUT=30)
    @patch("books.llm._prepare_image", return_value=("encoded-test-image", False))
    @patch("books.llm.urlopen")
    def test_request_uses_local_vision_model_and_validates_its_response(self, mock_urlopen, _prepare_image):
        from .llm import recommend_page_settings

        mock_urlopen.return_value = io.BytesIO(json.dumps({
            "message": {"content": json.dumps(self.recommendation())},
        }).encode("utf-8"))
        page = type("Page", (), {"book": type("Book", (), {"auto_crop_page": False, "language": "fra+eng"})()})()

        result = recommend_page_settings(page)

        self.assertEqual(result, self.recommendation())
        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(payload["model"], "vision-test")
        self.assertEqual(payload["messages"][1]["images"], ["encoded-test-image"])
        self.assertIn("deu", payload["format"]["properties"]["settings"]["properties"]["language"]["enum"])
        self.assertEqual(payload["format"]["properties"]["settings"]["properties"]["brightness"]["maximum"], 1.8)
        self.assertIn("fra+eng", payload["messages"][1]["content"])
        self.assertFalse(payload["stream"])
        mock_urlopen.assert_called_once_with(request, timeout=30)

    def test_image_is_resized_and_encoded_as_jpeg(self):
        import base64
        from .llm import _prepare_image

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "large.png"
            Image.new("RGB", (1600, 800), "white").save(path)
            page = type("Page", (), {
                "image": type("ImageFile", (), {"path": str(path)})(),
                "rotation": 0,
                "book": type("Book", (), {"auto_crop_page": False})(),
            })()

            encoded, yellow_paper = _prepare_image(page)

        with Image.open(io.BytesIO(base64.b64decode(encoded))) as prepared:
            self.assertEqual(prepared.format, "JPEG")
            self.assertEqual(prepared.size, (1280, 640))
        self.assertFalse(yellow_paper)

    def test_yellow_paper_is_detected_from_multiple_page_corners(self):
        import base64
        from .llm import _prepare_image

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "yellow.png"
            page = Image.new("RGB", (600, 800), (235, 224, 187))
            ImageDraw.Draw(page).text((80, 120), "Texte", fill=(55, 42, 30))
            page.save(path)
            source = type("Page", (), {
                "image": type("ImageFile", (), {"path": str(path)})(),
                "rotation": 0,
                "book": type("Book", (), {"auto_crop_page": False})(),
            })()

            _encoded, yellow_paper = _prepare_image(source)

        self.assertTrue(yellow_paper)

    def test_yellow_paper_detection_overrides_model_recommendation(self):
        from .llm import _validate_recommendation

        proposal = self.recommendation()
        proposal["settings"]["neutralize_yellow_background"] = False

        result = _validate_recommendation(proposal, yellow_paper_detected=True)

        self.assertTrue(result["settings"]["neutralize_yellow_background"])
        self.assertIn("papier jauni détecté", result["reason"])

    def test_out_of_range_model_recommendations_are_clamped_to_safe_steps(self):
        from .llm import _validate_recommendation

        proposal = self.recommendation()
        proposal["settings"].update({
            "brightness": 9,
            "contrast": 0.1,
            "show_through": -4,
            "unsharp": 1.26,
        })
        result = _validate_recommendation(proposal)

        self.assertEqual(result["settings"]["brightness"], 1.8)
        self.assertEqual(result["settings"]["contrast"], 0.5)
        self.assertEqual(result["settings"]["show_through"], 0.0)
        self.assertEqual(result["settings"]["unsharp"], 1.3)
        self.assertIn("ajustées aux plages sûres", result["reason"])

    def test_non_finite_model_recommendations_are_rejected(self):
        from .llm import OllamaRecommendationError, _validate_recommendation

        proposal = self.recommendation()
        proposal["settings"]["brightness"] = float("inf")
        with self.assertRaises(OllamaRecommendationError):
            _validate_recommendation(proposal)

    def test_recommendation_does_not_require_a_model_explanation(self):
        from .llm import _validate_recommendation

        proposal = self.recommendation()
        proposal.pop("reason")
        result = _validate_recommendation(proposal)

        self.assertEqual(result["settings"], self.recommendation()["settings"])
        self.assertIn("vérifiez l’aperçu", result["reason"])

    def test_long_model_explanation_is_bounded_not_rejected(self):
        from .llm import _validate_recommendation

        proposal = self.recommendation()
        proposal["reason"] = "Explication détaillée. " * 100

        result = _validate_recommendation(proposal)

        self.assertLessEqual(len(result["reason"]), 600)
        self.assertTrue(result["reason"].endswith("…"))

    @override_settings(OLLAMA_BASE_URL="http://127.0.0.1:11434", OLLAMA_TIMEOUT=5)
    @patch("books.llm._prepare_image", return_value=("encoded-test-image", False))
    @patch("books.llm.urlopen", side_effect=TimeoutError)
    def test_model_timeout_has_an_actionable_message(self, _mock_urlopen, _prepare_image):
        from .llm import OllamaRecommendationError, recommend_page_settings

        page = type("Page", (), {"book": type("Book", (), {
            "auto_crop_page": False, "language": "fra+eng",
        })()})()
        with self.assertRaisesRegex(OllamaRecommendationError, "5 secondes"):
            recommend_page_settings(page)


@override_settings(JOBS_RUN_INLINE=True)      # the tests run a job in the request instead of in a thread
class JobTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.owner = user_model.objects.create_user("worker", password="pw")
        self.other = user_model.objects.create_user("intruder", password="pw")
        self.media = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.media, ignore_errors=True)
        override = override_settings(MEDIA_ROOT=self.media)
        override.enable()
        self.addCleanup(override.disable)
        self.book = Book.objects.create(owner=self.owner, title="Livre", auto_crop_page=False)
        for order in (1, 2):
            PageImage.objects.create(book=self.book, order=order, original_name=f"p{order}.png",
                                     image=SimpleUploadedFile(f"p{order}.png", png_bytes(text_page(300, 400))))
        self.client.force_login(self.owner)

    def refreshed(self):
        self.book.refresh_from_db()
        return self.book

    @staticmethod
    def fake_pdf(toc=None, before=None):
        """A stand-in for build_searchable_pdf: writes a small PDF and reports progress."""
        def build(book, pages, output, progress=None):
            if before:
                before(progress)
            if progress:
                progress(0.5, "Reconnaissance du texte (OCR)…")
            Image.new("RGB", (100, 140), "white").save(output, "PDF")
            return toc or {}
        return build

    def test_processing_runs_in_a_job_and_leaves_the_pdf_and_a_message(self):
        toc = {"links": 3, "toc_pages": 1, "unresolved": 2}
        with patch("books.jobs.build_searchable_pdf", side_effect=self.fake_pdf(toc)):
            response = self.client.post(f"/livres/{self.book.pk}/traiter/", self.form_data(), follow=True)
        self.assertContains(response, "Traitement lancé en arrière-plan")
        self.assertContains(response, "PDF consultable créé avec succès.")           # the job's result, shown once
        self.assertContains(response, "3 lien(s) cliquable(s)")
        book = self.refreshed()
        self.assertEqual((book.status, book.job_kind, book.job_result), ("done", "", ""))
        self.assertTrue(book.output_pdf)
        self.assertNotContains(self.client.get(f"/livres/{self.book.pk}/"), "PDF consultable créé")   # not shown twice

    def test_a_failing_job_reports_the_error(self):
        with patch("books.jobs.build_searchable_pdf", side_effect=RuntimeError("tesseract a planté")):
            self.client.post(f"/livres/{self.book.pk}/traiter/", self.form_data())
        book = self.refreshed()
        self.assertEqual((book.status, book.job_kind), ("error", ""))
        self.assertIn("tesseract a planté", book.error_message)
        self.assertContains(self.client.get(f"/livres/{self.book.pk}/"), "L’opération a échoué")

    def test_a_job_can_be_cancelled(self):
        from books import jobs

        def ask_to_cancel(progress):
            self.client.post(f"/livres/{self.book.pk}/annuler/")      # the user presses "Annuler" mid-job
            progress(0.2, "encore une page")                          # the worker notices at its next report
        with patch("books.jobs.build_searchable_pdf", side_effect=self.fake_pdf(before=ask_to_cancel)):
            self.client.post(f"/livres/{self.book.pk}/traiter/", self.form_data())
        book = self.refreshed()
        self.assertEqual((book.status, book.job_kind, book.job_cancel), ("new", "", False))
        self.assertFalse(book.output_pdf)
        self.assertContains(self.client.get(f"/livres/{self.book.pk}/"), "Opération annulée.")

    def test_analysis_runs_in_a_job_too(self):
        response = self.client.post(f"/livres/{self.book.pk}/analyser/", follow=True)
        self.assertContains(response, "Analyse terminée : 2 page(s) de texte seulement")
        self.assertEqual(set(self.book.pages.values_list("page_type", flat=True)), {"text"})
        self.assertEqual(self.refreshed().job_kind, "")

    def test_a_busy_book_refuses_changes_and_a_second_job(self):
        from django.utils import timezone
        Book.objects.filter(pk=self.book.pk).update(job_kind="process", status="processing", job_heartbeat=timezone.now())
        base = f"/livres/{self.book.pk}"
        for path in ("/traiter/", "/analyser/", "/parametres/", "/supprimer/"):
            response = self.client.post(base + path, self.form_data(), follow=True)
            self.assertContains(response, "Un traitement est en cours", msg_prefix=path)
        self.assertTrue(Book.objects.filter(pk=self.book.pk).exists())
        upload = self.client.post(base + "/importer/", {"images": SimpleUploadedFile("n.png", png_bytes(text_page(100, 100)))}, follow=True)
        self.assertContains(upload, "Un traitement est en cours")
        self.assertEqual(self.book.pages.count(), 2)
        page = self.book.pages.first()
        for suffix, data in (
            (f"/pages/{page.pk}/pivoter/", {}),
            (f"/pages/{page.pk}/photo/", {"value": "1"}),
            (f"/pages/{page.pk}/rognage/", {"custom": "1", "left": "5"}),
            (f"/pages/{page.pk}/type/", {"type": "photo"}),
        ):
            response = self.client.post(base + suffix, data)
            self.assertEqual(response.status_code, 409, msg=suffix)
            self.assertIn("Un traitement est en cours", response.json()["error"])
        page.refresh_from_db()
        self.assertEqual((page.rotation, page.has_photo, page.trim_custom, page.page_type), (0, False, False, ""))

    def test_a_job_that_lost_its_worker_is_closed_with_an_explanation(self):
        from datetime import timedelta
        from django.utils import timezone
        Book.objects.filter(pk=self.book.pk).update(job_kind="process", status="processing",
                                                    job_heartbeat=timezone.now() - timedelta(minutes=10))
        page = self.client.get(f"/livres/{self.book.pk}/")
        self.assertContains(page, "interrompu")
        book = self.refreshed()
        self.assertEqual((book.status, book.job_kind), ("error", ""))

    def test_progress_endpoint_reports_the_job_and_belongs_to_the_owner(self):
        from django.utils import timezone
        Book.objects.filter(pk=self.book.pk).update(job_kind="process", status="processing", job_progress=425,
                                                    job_step="Page 3 sur 7", job_heartbeat=timezone.now())
        data = self.client.get(f"/livres/{self.book.pk}/progression/").json()
        self.assertEqual((data["active"], data["percent"], data["step"], data["kind"]), (True, 42.5, "Page 3 sur 7", "process"))
        self.client.post(f"/livres/{self.book.pk}/annuler/")
        self.assertTrue(self.client.get(f"/livres/{self.book.pk}/progression/").json()["cancelling"])
        self.client.force_login(self.other)
        self.assertEqual(self.client.get(f"/livres/{self.book.pk}/progression/").status_code, 404)
        self.assertEqual(self.client.post(f"/livres/{self.book.pk}/annuler/").status_code, 404)
        self.client.force_login(self.owner)
        Book.objects.filter(pk=self.book.pk).update(job_kind="", job_cancel=False)
        self.assertFalse(self.client.get(f"/livres/{self.book.pk}/progression/").json()["active"])

    def test_the_page_shows_the_progress_bar_only_while_a_job_runs(self):
        from django.utils import timezone
        self.assertContains(self.client.get(f"/livres/{self.book.pk}/"), 'id="job-panel"')
        self.assertContains(self.client.get(f"/livres/{self.book.pk}/"), " hidden>")
        Book.objects.filter(pk=self.book.pk).update(job_kind="analyze", job_progress=300, job_heartbeat=timezone.now())
        html = self.client.get(f"/livres/{self.book.pk}/").content.decode()
        panel = html[html.index('id="job-panel"'):][:400]
        self.assertNotIn(" hidden>", panel.split(">")[0] + ">")
        self.assertRegex(html, r'<select id="page-type"[^>]* disabled>')
        self.assertIn('id="photo-toggle" disabled', html)

    def form_data(self):
        """What the settings form of the page would post for this book."""
        from django.forms import CheckboxInput
        from .forms import ProcessingSettingsForm
        data = {}
        for name, field in ProcessingSettingsForm().fields.items():
            value = getattr(self.book, name)
            if isinstance(field.widget, CheckboxInput):
                if value:
                    data[name] = "on"
            else:
                data[name] = str(value)
        return data


class ViewTests(TestCase):
    def setUp(self):
        self.media = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.media, ignore_errors=True)
        override = override_settings(MEDIA_ROOT=self.media)
        override.enable()
        self.addCleanup(override.disable)
        user_model = get_user_model()
        self.owner = user_model.objects.create_user("owner", password="pw")
        self.other = user_model.objects.create_user("other", password="pw")
        self.book = Book.objects.create(owner=self.owner, title="Livre")
        self.page = PageImage.objects.create(
            book=self.book, order=1, original_name="a.png",
            image=SimpleUploadedFile("a.png", png_bytes(text_page(300, 400))),
        )

    def test_pages_require_login(self):
        response = self.client.get(f"/livres/{self.book.pk}/")
        self.assertEqual(response.status_code, 302)
        self.assertIn("login", response["Location"])

    def test_book_page_and_preview_work_for_owner(self):
        self.client.force_login(self.owner)
        page = self.client.get(f"/livres/{self.book.pk}/")
        self.assertEqual(page.status_code, 200)
        self.assertContains(page, "Proposer des réglages avec l’IA")
        preview = self.client.get(f"/livres/{self.book.pk}/apercu/{self.page.pk}/", {
            "language": "fra", "page_format": "a4", "a4_margin_mm": 5, "a4_align": "center",
            "output_mode": "color", "brightness": 1, "contrast": 1,
            "unsharp": 0, "show_through": 0, "denoise_mode": "off", "crop_padding": 20,
            "photo_output_mode": "color", "photo_brightness": 1.3, "photo_contrast": 1.1,
            "photo_unsharp": 0.3, "photo_show_through": 0, "photo_denoise_mode": "off",
        })
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview["Content-Type"], "application/json")
        self.assertEqual(set(preview.json()), {"adjusted", "rendered"})
        with patch("books.views.separate_text_background", return_value=(
            Image.new("RGB", (100, 120), "white"),
            Image.new("RGB", (100, 120), "white"),
            Image.new("RGB", (100, 120), "white"),
        )):
            layers = self.client.get(
                f"/livres/{self.book.pk}/apercu/{self.page.pk}/",
                {"language": "fra", "page_format": "original", "layers": "1", "size": 600},
            )
        self.assertEqual(layers.status_code, 200)
        self.assertEqual(layers["Content-Type"], "application/json")
        self.assertEqual(set(layers.json()), {"adjusted", "rendered", "text", "background", "overlay"})

    def test_preview_layers_report_unavailable_paddle_ocr(self):
        self.client.force_login(self.owner)
        with patch("books.views.separate_text_background",
                   side_effect=TextLayerError("PaddleOCR est indisponible.")):
            response = self.client.get(
                f"/livres/{self.book.pk}/apercu/{self.page.pk}/",
                {"language": "fra", "page_format": "original", "layers": "1"},
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"error": "PaddleOCR est indisponible."})

    @patch("books.views.recommend_page_settings")
    def test_ai_recommendation_is_not_saved_and_is_scoped_to_book_owner(self, recommend):
        self.client.force_login(self.owner)
        proposal = OllamaRecommendationTests().recommendation()
        recommend.return_value = proposal
        url = f"/livres/{self.book.pk}/reglages-ia/{self.page.pk}/"

        response = self.client.post(url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), proposal)
        self.book.refresh_from_db()
        self.assertEqual(self.book.brightness, 1.0)
        self.client.force_login(self.other)
        self.assertEqual(self.client.post(url).status_code, 404)
        self.assertEqual(recommend.call_count, 1)

    def test_ai_recommendation_reports_local_model_errors(self):
        from .llm import OllamaRecommendationError

        self.client.force_login(self.owner)
        with patch("books.views.recommend_page_settings", side_effect=OllamaRecommendationError("Ollama indisponible.")):
            response = self.client.post(f"/livres/{self.book.pk}/reglages-ia/{self.page.pk}/")
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json(), {"error": "Ollama indisponible."})

    def test_global_prices_are_saved_from_home_and_used_for_book_yield(self):
        from decimal import Decimal

        self.client.force_login(self.owner)
        home = self.client.get("/")
        self.assertContains(home, "Tarifs globaux")
        self.assertContains(home, 'name="price_text"')
        self.page.page_type = "text"
        self.page.save(update_fields=["page_type"])
        response = self.client.post("/", {
            "price_text": "12.50",
            "price_table": "20",
            "price_chart": "25",
            "price_photo": "30",
            "currency": "MGA",
        })
        self.assertRedirects(response, "/")

        prices = PriceSettings.load()
        self.assertEqual(prices.price_text, Decimal("12.50"))
        self.assertEqual(prices.currency, "MGA")
        self.assertEqual(self.book.yield_summary()["total"], Decimal("0.00"))

        detail = self.client.get(f"/livres/{self.book.pk}/")
        self.assertContains(detail, "€")
        self.assertContains(detail, 'name="price_text"')

        from .forms import ProcessingSettingsForm
        settings_data = ProcessingSettingsForm(instance=self.book).initial.copy()
        settings_data.update({
            "price_text": "4.75",
            "price_table": "0",
            "price_chart": "0",
            "price_photo": "0",
            "currency": "€",
        })
        response = self.client.post(f"/livres/{self.book.pk}/parametres/", settings_data)
        self.assertRedirects(response, f"/livres/{self.book.pk}/")
        self.book.refresh_from_db()
        self.assertEqual(self.book.yield_summary()["total"], Decimal("4.75"))

        create_response = self.client.post("/livres/nouveau/", {"title": "Nouveau livre"})
        self.assertEqual(create_response.status_code, 302)
        new_book = Book.objects.get(title="Nouveau livre")
        self.assertEqual(new_book.price_text, Decimal("12.50"))
        self.assertEqual(new_book.currency, "MGA")

    def test_other_users_cannot_see_or_change_a_book(self):
        self.client.force_login(self.other)
        self.assertEqual(self.client.get(f"/livres/{self.book.pk}/").status_code, 404)
        self.assertEqual(self.client.post(f"/livres/{self.book.pk}/pages/{self.page.pk}/pivoter/", {"direction": "cw"}).status_code, 404)

    def test_media_files_are_served_only_to_their_owner(self):
        url = f"/media/{self.page.image.name}"
        self.client.force_login(self.owner)
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        response.close()
        self.client.force_login(self.other)
        self.assertEqual(self.client.get(url).status_code, 404)

    @override_settings(JOBS_RUN_INLINE=True)
    def test_analysis_sets_page_types_and_counts_them(self):
        self.client.force_login(self.owner)
        response = self.client.post(f"/livres/{self.book.pk}/analyser/", follow=True)
        self.assertEqual(response.status_code, 200)
        self.page.refresh_from_db()
        self.assertEqual(self.page.page_type, "text")
        self.assertFalse(self.page.has_photo)
        self.assertContains(response, "1 page(s) de texte")
        self.client.force_login(self.other)
        self.assertEqual(self.client.post(f"/livres/{self.book.pk}/analyser/").status_code, 404)

    def test_each_page_can_have_its_own_edge_trim(self):
        self.client.force_login(self.owner)
        url = f"/livres/{self.book.pk}/pages/{self.page.pk}/rognage/"
        reply = self.client.post(url, {"custom": "1", "left": "2", "top": "1", "right": "7.5", "bottom": "99"}).json()
        self.assertEqual(reply, {"custom": True, "left": 2.0, "top": 1.0, "right": 7.5, "bottom": 25.0})   # capped at 25%
        self.page.refresh_from_db()
        self.assertEqual(self.page.custom_trim, (2.0, 1.0, 7.5, 25.0))
        self.client.post(url, {"custom": "0", "left": "2", "top": "1", "right": "7.5", "bottom": "3"})
        self.page.refresh_from_db()
        self.assertIsNone(self.page.custom_trim)                       # back to the book's trim
        self.client.force_login(self.other)
        self.assertEqual(self.client.post(url, {"custom": "1"}).status_code, 404)

    def test_page_trim_replaces_the_book_trim_for_that_page_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "p.png"
            Image.new("RGB", (1000, 2000), "white").save(source)
            book = Book(auto_rotate=False, auto_crop_page=False, page_format="original", crop_margins=False,
                        deskew=False, edge_trim_left=10, edge_trim_right=10)
            self.assertEqual(process_image(source, book).size, (800, 2000))                          # the book's trim
            self.assertEqual(process_image(source, book, page_trim=(0, 0, 0, 0)).size, (1000, 2000))  # this page: none
            self.assertEqual(process_image(source, book, page_trim=(5, 0, 5, 10)).size, (900, 1800))  # this page: its own

    def test_rotation_and_photo_marking_are_saved_per_page(self):
        self.client.force_login(self.owner)
        base = f"/livres/{self.book.pk}/pages/{self.page.pk}"
        self.assertEqual(self.client.post(f"{base}/pivoter/", {"direction": "cw"}).json(), {"rotation": 90})
        self.assertEqual(self.client.post(f"{base}/pivoter/", {"direction": "ccw"}).json(), {"rotation": 0})
        self.assertEqual(self.client.post(f"{base}/photo/", {"value": "1"}).json(), {"has_photo": True})
        self.page.refresh_from_db()
        self.assertTrue(self.page.has_photo)
