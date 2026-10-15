from django.urls import path
from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("livres/nouveau/", views.create_book, name="create_book"),
    path("livres/<int:pk>/", views.book_detail, name="book_detail"),
    path("livres/<int:pk>/importer/", views.upload_pages, name="upload_pages"),
    path("livres/<int:pk>/apercu/<int:page_pk>/", views.preview_page, name="preview_page"),
    path("livres/<int:pk>/pages/<int:page_pk>/pivoter/", views.rotate_page, name="rotate_page"),
    path("livres/<int:pk>/pages/<int:page_pk>/photo/", views.toggle_photo, name="toggle_photo"),
    path("livres/<int:pk>/parametres/", views.save_settings, name="save_settings"),
    path("livres/<int:pk>/traiter/", views.process_book, name="process_book"),
    path("livres/<int:pk>/telecharger/", views.download_pdf, name="download_pdf"),
    path("livres/<int:pk>/supprimer/", views.delete_book, name="delete_book"),
]
