from django.db import migrations


def initialize_unpriced_books(apps, schema_editor):
    Book = apps.get_model('books', 'Book')
    PriceSettings = apps.get_model('books', 'PriceSettings')
    prices = PriceSettings.objects.filter(pk=1).values(
        'price_text', 'price_table', 'price_chart', 'price_photo', 'currency',
    ).first()
    if not prices:
        return
    Book.objects.filter(
        price_text=0,
        price_table=0,
        price_chart=0,
        price_photo=0,
    ).update(**prices)


class Migration(migrations.Migration):

    dependencies = [
        ('books', '0020_book_currency_book_price_chart_book_price_photo_and_more'),
    ]

    operations = [
        migrations.RunPython(initialize_unpriced_books, migrations.RunPython.noop),
    ]
