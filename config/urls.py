from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views
from books.views import media_file

urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/login/", auth_views.LoginView.as_view(template_name="registration/login.html"), name="login"),
    path("accounts/logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("accounts/", include("django.contrib.auth.urls")),
    # Uploaded scans and generated PDFs are served only to the user who owns them.
    path("media/<path:path>", media_file, name="media_file"),
    path("", include("books.urls")),
]
