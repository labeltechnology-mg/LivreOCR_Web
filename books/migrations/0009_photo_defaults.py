from django.db import migrations, models


def update_untouched_photo_settings(apps, schema_editor):
    """Books created before this change still hold the old neutral photo values: move them to
    the new defaults (they never had a photo page, so nobody chose these numbers)."""
    Book = apps.get_model("books", "Book")
    Book.objects.filter(photo_brightness=1.0, photo_contrast=1.0).update(photo_brightness=1.3, photo_contrast=1.1)


class Migration(migrations.Migration):

    dependencies = [
        ('books', '0008_photo_pages'),
    ]

    operations = [
        migrations.AlterField(
            model_name='book',
            name='photo_brightness',
            field=models.FloatField(default=1.3),
        ),
        migrations.AlterField(
            model_name='book',
            name='photo_contrast',
            field=models.FloatField(default=1.1),
        ),
        migrations.RunPython(update_untouched_photo_settings, migrations.RunPython.noop),
    ]
