import io
import shutil
import tempfile
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
            "language": "fra", "page_format": "a4", "output_mode": "color", "brightness": 1, "contrast": 1,
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

    def test_rotation_and_photo_marking_are_saved_per_page(self):
        self.client.force_login(self.owner)
        base = f"/livres/{self.book.pk}/pages/{self.page.pk}"
        self.assertEqual(self.client.post(f"{base}/pivoter/", {"direction": "cw"}).json(), {"rotation": 90})
        self.assertEqual(self.client.post(f"{base}/pivoter/", {"direction": "ccw"}).json(), {"rotation": 0})
        self.assertEqual(self.client.post(f"{base}/photo/", {"value": "1"}).json(), {"has_photo": True})
        self.page.refresh_from_db()
        self.assertTrue(self.page.has_photo)
