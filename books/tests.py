import io
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from PIL import Image, ImageDraw

from .image_processing import (
    A4_PX, detect_page_quad, estimate_skew, fit_to_a4, process_image, reduce_show_through,
)
from .models import Book, PageImage
from .orientation import find_tesseract

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


class ImageProcessingTests(TestCase):
    def test_a4_sheet_follows_page_orientation(self):
        self.assertEqual(fit_to_a4(Image.new("RGB", (900, 1400))).size, A4_PX)
        self.assertEqual(fit_to_a4(Image.new("RGB", (1400, 900))).size, A4_PX[::-1])

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
        self.assertEqual(self.client.get(f"/livres/{self.book.pk}/").status_code, 200)
        preview = self.client.get(f"/livres/{self.book.pk}/apercu/{self.page.pk}/", {
            "language": "fra", "page_format": "a4", "a4_margin_mm": 5, "a4_align": "center",
            "output_mode": "color", "brightness": 1, "contrast": 1,
            "unsharp": 0, "show_through": 0, "denoise_mode": "off", "crop_padding": 20,
            "photo_output_mode": "color", "photo_brightness": 1.3, "photo_contrast": 1.1,
            "photo_unsharp": 0.3, "photo_show_through": 0, "photo_denoise_mode": "off",
        })
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview["Content-Type"], "image/jpeg")

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
