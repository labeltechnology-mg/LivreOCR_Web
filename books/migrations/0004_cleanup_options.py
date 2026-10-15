from django.db import migrations, models


def convert_old_settings(apps, schema_editor):
    Book = apps.get_model("books", "Book")
    # The old boolean denoise used an edge-preserving filter; sharpness 1.0 meant "off".
    Book.objects.filter(denoise=True).update(denoise_mode="edge")
    for book in Book.objects.exclude(sharpness=1.0):
        book.unsharp = round(max(0.0, min(2.0, (book.sharpness - 1.0) * 0.8)), 2)
        book.save(update_fields=["unsharp"])


class Migration(migrations.Migration):

    dependencies = [
        ("books", "0003_page_format"),
    ]

    operations = [
        migrations.AddField("book", "clahe", models.BooleanField(default=False)),
        migrations.AddField("book", "unsharp", models.FloatField(default=0.0)),
        migrations.AddField("book", "denoise_mode", models.CharField(
            choices=[("off", "Aucun"), ("median", "Médian léger"), ("edge", "Préservant les contours")],
            default="off", max_length=10)),
        migrations.RunPython(convert_old_settings, migrations.RunPython.noop),
        migrations.RemoveField("book", "sharpness"),
        migrations.RemoveField("book", "denoise"),
    ]
