from django.contrib import admin
from .models import Book, PageImage

class PageImageInline(admin.TabularInline):
    model = PageImage
    extra = 0

@admin.register(Book)
class BookAdmin(admin.ModelAdmin):
    list_display = ("title", "owner", "status", "created_at")
    list_filter = ("status", "created_at")
    search_fields = ("title", "owner__username")
    inlines = [PageImageInline]

@admin.register(PageImage)
class PageImageAdmin(admin.ModelAdmin):
    list_display = ("book", "original_name", "order", "uploaded_at")
