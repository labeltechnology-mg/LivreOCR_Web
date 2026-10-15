# Ocypete OCR — application multi-utilisateurs (prototype)

<p align="center"><img src="books/static/books/img/ocypete-logo.png" alt="Logo Ocypete OCR" width="220"></p>

> Ocypete OCR est le nom de l'application. Le dépôt, les variables d'environnement (`LIVREOCR_MEDIA_ROOT`) et les noms de code gardent le préfixe historique « livreocr ».

Application Django en français pour gérer des livres numérisés, importer des photos/scans, prétraiter les pages et créer un PDF avec une couche texte OCR.

## Architecture
- Django : comptes, sessions, vues, formulaires et administration.
- SQLite en développement ; PostgreSQL possible via `DATABASE_URL`.
- Modèles `Book` et `PageImage`, chaque livre étant associé à son propriétaire.
- OpenCV/Pillow : rotation EXIF, contraste, luminosité, recadrage conservateur.
- OCRmyPDF + Tesseract : PDF consultable avec images de page préservées.
- Fichiers stockés localement dans `media/` par défaut.

## Prérequis
- Python 3.10+
- Tesseract OCR et les données linguistiques voulues (`fra`, `eng`, etc.)
- OCRmyPDF et ses dépendances système

Consulter la documentation OCRmyPDF pour l'installation selon le système :
https://ocrmypdf.readthedocs.io/en/stable/installation.html

Exemple Ubuntu/Debian :
```bash
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng ocrmypdf
```

## Installation locale
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```
Ouvrez http://127.0.0.1:8000/ et connectez-vous avec le compte créé. Les comptes peuvent être créés depuis `/admin/` par un administrateur.

## Production : configuration indispensable
Ne déployez pas avec les paramètres de développement. Définissez au minimum :
- `DJANGO_SECRET_KEY` : secret long et aléatoire
- `DJANGO_DEBUG=0`
- `DJANGO_ALLOWED_HOSTS=example.com,www.example.com`
- `DATABASE_URL=postgres://user:password@host:5432/database`
- `LIVREOCR_MEDIA_ROOT` si vous utilisez un stockage local persistant

Puis :
```bash
python manage.py check --deploy
python manage.py collectstatic
gunicorn config.wsgi:application
```
Servez les médias via un stockage privé ou une couche de téléchargement authentifiée. Pour une vraie production, configurez HTTPS, sauvegardes, limites de débit, quotas, antivirus de fichiers, journaux et politique de conservation.

## Docker
L'image contient Tesseract (fra, eng, deu, lat, spa, por, osd et l'allemand gothique `deu_latf`), Ghostscript et ocrmypdf : rien d'autre à installer.

```bash
cp .env.example .env        # renseigner DJANGO_SECRET_KEY et POSTGRES_PASSWORD
docker compose up --build
docker compose exec web python manage.py createsuperuser
```
Ouvrez http://localhost:8000/. Les scans et les PDF sont dans le volume `media-data`, la base dans `db-data` : `docker compose down` les conserve, `docker compose down -v` les supprime.

- `DJANGO_HTTPS=0` (valeur du `.env.example`) sert uniquement à tester en http sur localhost. Derrière un reverse proxy HTTPS, mettez `DJANGO_HTTPS=1` et renseignez `DJANGO_ALLOWED_HOSTS` et `DJANGO_CSRF_TRUSTED_ORIGINS` avec votre domaine.
- L'OCR s'exécute dans la requête : `GUNICORN_TIMEOUT` (900 s par défaut) doit dépasser la durée du plus gros livre.
- Les images de pages et les PDF ne sont servis qu'à leur propriétaire (route `/media/…` authentifiée).

## Workflow Git
- `main` est toujours déployable. Travaillez sur une branche (`feature/…`, `fix/…`) puis ouvrez une Pull Request vers `main` (ou `develop` si vous l'utilisez).
- À chaque push et chaque PR, GitHub Actions (`.github/workflows/ci.yml`) lance : `manage.py check`, vérification des migrations, `check --deploy`, les tests, puis la construction de l'image Docker et la présence des langues OCR.
- Une PR ne se fusionne que si la CI est verte. Activez la protection de branche `main` dans *Settings → Branches* (« Require status checks » : *Checks and tests*, *Docker image builds*).
- Pour publier une version : `git tag v1.0.0 && git push --tags`. Le workflow `docker-publish.yml` pousse l'image sur `ghcr.io/<compte>/<dépôt>` (tags `1.0.0`, `1.0` et `latest`).
- Dependabot propose chaque semaine les mises à jour pip, Docker et GitHub Actions.
- Tests en local : `python manage.py test`.

## Limites connues du prototype
1. Le traitement OCR est synchrone dans la requête HTTP : pour les livres volumineux, ajoutez une file de tâches (Celery/RQ) et un worker.
2. Les pages sont classées par ordre d’import, pas par détection automatique du contenu ; utilisez des noms séquentiels et contrôlez l’ordre.
3. Le recadrage automatique par seuil peut couper du contenu de bord ou mal traiter des pages jaunies. Vérifiez les pages avant usage archivistique.
4. Les images sont privées via les vues applicatives, mais les fichiers média ne doivent pas être servis publiquement sans contrôle d'accès en production.
5. La suppression de projet doit aussi nettoyer les fichiers physiques dans une version durcie.
6. Le bouton de traitement réutilise les paramètres enregistrés. En environnement multi-utilisateur, ne publiez jamais `media/` en accès direct public.
7. Le code de démarrage doit faire l'objet de tests et d'un audit de sécurité avant utilisation avec des documents sensibles.

## Prochaines étapes conseillées
- Traitement asynchrone Celery + Redis avec progression en base
- Réorganisation manuelle des pages (drag-and-drop)
- Aperçu image avant/après et correction manuelle de l’orientation
- Stockage objet privé (S3 compatible) et quotas par utilisateur
- Tests automatisés et suivi des erreurs
