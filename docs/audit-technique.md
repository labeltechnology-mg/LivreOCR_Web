# Audit technique — Ocypete OCR (LivreOCR_Web)

Audit réalisé le 16/10/2026 sur `main` @ `9ed3797`, arbre de travail propre. Aucun fichier de code n'a été modifié pendant l'audit.

## 1. État initial vérifié

| Vérification | Résultat |
|---|---|
| `manage.py check` | OK, aucun problème |
| `makemigrations --check` | OK, migrations à jour (0001 → 0013) |
| `check --deploy` (DEBUG=0) | 1 avertissement : `SECURE_HSTS_SECONDS` non défini |
| `manage.py test` | 21 tests, **20 OK, 1 erreur** : `TocEndToEndTests` échoue en local car Tesseract n'a que `afr` et `osd` (pas `fra`). Problème d'environnement, pas de code ; la CI installe `fra`/`eng`. |
| Environnement local | Python 3.14, Django 5.2.18, Pillow 12.3, OpenCV 5.0, NumPy 2.5, ocrmypdf 17.13, pikepdf 10.17 |
| Données locales | 1 utilisateur, 1 livre, 0 page, pas de dossier `media/` |

### Problème de sécurité bloquant

Le fichier **`.env ` (avec un espace final)** est versionné depuis le commit `9ed3797` et poussé sur `origin` (GitHub). Le motif `.env` du `.gitignore` ne le couvre pas à cause de l'espace. Il contient une **`DJANGO_SECRET_KEY` réelle** (67 caractères).

Actions nécessaires (à valider, rien n'a été fait) :
1. Considérer la clé comme compromise et en **générer une nouvelle** sur tout environnement qui l'utilise.
2. `git rm --cached ".env "` puis ajouter un motif qui couvre la variante (`.env*` avec exception `!.env.example`).
3. Réécrire l'historique (`git filter-repo`) seulement si le dépôt est public ou partagé largement. La rotation de la clé reste obligatoire dans tous les cas.

## 2. Architecture détectée

- **Projet** `config/` + **une seule application** `books/`. Pas de DRF, pas de serializers, pas d'API JSON en dehors de 2 endpoints `fetch`.
- **Rendu** : Django Templates. `base.html` contient un en-tête horizontal (logo + déconnexion). **Il n'existe pas de sidebar de navigation.** Le panneau latéral de `book_detail.html` est une « boîte à outils » de réglages, pas une navigation.
- **JS** : vanilla, inline dans `book_detail.html` (≈250 lignes) : aperçu, zoom synchronisé, rotation, préréglages, contrôle d'import. Pas de fichier JS statique.
- **CSS** : un seul fichier `books/static/books/css/app.css` (235 lignes), avec des variables `:root` et un cache-busting par `asset_version`.
- **Base** : SQLite en dev, PostgreSQL via `DATABASE_URL` (Docker Compose).
- **Stockage** : `FileSystemStorage`, `MEDIA_ROOT` configurable, fichiers servis **uniquement au propriétaire** via la vue `media_file` (testée).
- **Asynchrone** : **aucun**. L'OCR tourne dans la requête HTTP, avec un timeout gunicorn de 900 s.
- **Déploiement** : Dockerfile (Tesseract avec fra, eng, deu, lat, spa, por, osd, deu_latf, plus Ghostscript), Compose (web + postgres), CI GitHub Actions (checks, tests, build de l'image), publication GHCR sur tag.

### Routes (`books/urls.py`)

| Route | Vue | Rôle |
|---|---|---|
| `/` | `dashboard` | Liste des livres et 3 compteurs |
| `/livres/nouveau/` | `create_book` | Création (titre seul) |
| `/livres/<pk>/` | `book_detail` | Espace de travail unique : réglages, aperçu, import |
| `/livres/<pk>/importer/` | `upload_pages` | Import multiple |
| `/livres/<pk>/apercu/<page>/` | `preview_page` | Rendu JPEG à la volée avec les réglages en query string, sans sauvegarde |
| `/livres/<pk>/pages/<page>/pivoter/` | `rotate_page` | Rotation manuelle de 90° (JSON) |
| `/livres/<pk>/pages/<page>/photo/` | `toggle_photo` | Bascule « page avec photo » (JSON) |
| `/livres/<pk>/analyser/` | `analyze_pages` | Classification texte, graphique ou image |
| `/livres/<pk>/parametres/` | `save_settings` | Enregistrement des réglages |
| `/livres/<pk>/traiter/` | `process_book` | Pipeline complet et OCR (synchrone) |
| `/livres/<pk>/telecharger/` | `download_pdf` | Téléchargement du PDF |
| `/livres/<pk>/supprimer/` | `delete_book` | Suppression |
| `/media/<path>` | `media_file` | Médias réservés au propriétaire |

Toutes les vues sont protégées par `@login_required` et filtrées par `owner=request.user` (vérifié). Les actions modifiantes passent par `@require_POST` et la protection CSRF.

## 3. Modèles

**`Book`** (≈35 champs) : propriétaire, titre, statut (`new`, `processing`, `done`, `error`), langue OCR, **tous les réglages de rendu au niveau du livre** (luminosité, contraste, CLAHE, nettoyage du fond, verso, débruitage, netteté, mode couleur/gris/N&B), un **second jeu `photo_*`** pour les pages marquées photo, le rognage manuel des 4 bords en %, le format A4 ou d'origine, la marge et l'alignement A4, la TDM cliquable, les options de recadrage et de rotation, **un seul `output_pdf`**, `error_message`.

**`PageImage`** : livre, image (originale, jamais modifiée), nom d'origine, `order` (unique par livre), `rotation` manuelle, `has_photo`, `page_type` (`text`, `chart`, `photo` ou vide).

Ce qui **n'existe pas** : blocs OCR, texte OCR stocké, éléments détectés avec coordonnées, images sources et occurrences, paramètres de recadrage par page, historique des exports, polices, tâches de traitement.

## 4. Traitement d'images (`image_processing.py`, `orientation.py`, `classify.py`)

Le pipeline `process_image()` est **non destructif** : il part toujours de l'original et rien n'est réécrit sur disque. L'aperçu et le PDF utilisent la même fonction, donc l'aperçu est fidèle.

1. Rotation EXIF.
2. `crop_to_page` : détection de la feuille sur un fond sombre (Otsu, morphologie, `minAreaRect`) et **correction de perspective** sur 4 coins. Les pages trop sombres utilisent la position médiane des pages voisines du même côté (cache). Le bord des pages voisines est rogné si la page est sombre.
3. Rotation automatique 0/90/180/270 : OSD Tesseract, puis vérification par lecture (`--psm 6`), avec cache.
4. Rognage manuel des 4 bords en % (réglage **global au livre**).
5. Luminosité et contraste (PIL), CLAHE sur L de Lab (préserve les noirs), nettoyage du fond (normalisation de l'éclairage par canal), débruitage (médian ou bilatéral), atténuation du verso.
6. Redressement par profil de projection (±5°).
7. Netteté (masque flou), puis gris ou N&B (Sauvola) **seulement si choisi**.
8. Recadrage conservateur des marges, puis placement sur A4 à 300 dpi.

Classification (`classify.py`) : heuristiques OpenCV qui distinguent texte, graphique et photo à l'échelle de **la page**. Les zones ne sont pas localisées et les coordonnées ne sont pas enregistrées.

## 5. OCR et PDF (`ocr.py`, `toc.py`)

- `build_searchable_pdf` : traite chaque page en PNG 300 dpi, assemble un PDF d'images (Pillow), puis lance **`ocrmypdf -l <langue> -O 1`** en sous-processus. Le résultat est **déjà un « PDF fac-similé avec couche OCR invisible »** (mode A de la cible).
- `toc.py` : détection de la table des matières, lecture des numéros de page imprimés (Tesseract en TSV, `psm 6`), puis ajout de liens et de signets avec **pikepdf**. En cas d'échec, l'erreur est absorbée et le PDF est conservé.
- **Le texte OCR n'est stocké nulle part dans l'application.** Il n'existe que dans le PDF final. Il n'y a donc ni correction manuelle ni mode reconstruit possible aujourd'hui.

## 6. Problèmes constatés

| # | Gravité | Problème |
|---|---|---|
| 1 | **Critique** | Secret Django versionné et poussé (voir §1). |
| 2 | Haute | OCR synchrone dans la requête : un livre volumineux bloque un worker gunicorn jusqu'à 15 min. Si le processus est tué, le livre reste bloqué en `processing`. |
| 3 | Haute | Pas de protection contre un double lancement (seulement un verrou JS côté client). |
| 4 | Moyenne | `delete_book` ne supprime pas les fichiers physiques (pages et PDF). Un nouveau traitement laisse l'ancien PDF orphelin. Le README le signale (limite 5). |
| 5 | Moyenne | Impossible de supprimer ou de réordonner une page après l'import. |
| 6 | Moyenne | Rognage et recadrage **globaux au livre**, pas par page. Pas d'édition manuelle des 4 coins. |
| 7 | Moyenne | Le texte OCR n'est pas conservé, ce qui bloque la correction, le mode B et la relance par zone. |
| 8 | Basse | Pas de limite en pixels à l'import (bombe de décompression : seul l'avertissement Pillow par défaut s'applique). Pas de détection de doublons. |
| 9 | Basse | Une seule ligne de JS par vue, inline et volumineuse. À extraire en fichiers statiques avant d'ajouter des éditeurs. |
| 10 | Basse | `MEDIA_URL` et `STATIC_URL` sont définis deux fois dans `settings.py`. `HSTS` n'est pas configuré. |
| 11 | Info | Tesseract local sans `fra`/`eng`, donc un test échoue en local (`sudo pacman -S tesseract-data-fra tesseract-data-eng`). |

## 7. À réutiliser tel quel

- Le pipeline `process_image` (non destructif, aperçu identique au rendu final) : on l'étend, on ne le remplace pas.
- `detect_page_quad` et `crop_to_page` : leurs 4 coins servent de **valeur initiale** au recadrage manuel.
- `apply_clahe` (Lab), `clean_background`, `reduce_show_through`, `binarize` : base des profils couleur.
- ocrmypdf + pikepdf + TDM : c'est le **mode A**. On le conserve.
- La vue `media_file`, le filtrage par propriétaire et les tests de permissions.
- `classify.py` : il sert au choix du profil par type de page.

## 8. Dépendances

| Dépendance | Décision |
|---|---|
| Django, Pillow, OpenCV, NumPy, psycopg, gunicorn, whitenoise, ocrmypdf, pikepdf, pdfminer.six | **Conserver** |
| Moteur PDF du mode B | **À ajouter.** Recommandé : **ReportLab** (BSD), qui gère le texte vectoriel, l'intégration des polices TTF et l'Unicode. PyMuPDF est écarté par défaut : licence **AGPL** (ou licence commerciale), contraignante pour un service web propriétaire. |
| Extraction OCR structurée | **Rien à ajouter** : appeler `tesseract … hocr/tsv` en sous-processus, comme le font déjà `orientation.py` et `toc.py`. `pytesseract` n'apporterait rien. |
| File de tâches | **À décider** (voir §10) |
| Recadrage JS | Pas de bibliothèque : 4 poignées sur un `<canvas>`/SVG suffisent et évitent une dépendance. |

## 9. Risques de régression

- **Migrations** : uniquement additives (nouveaux modèles, champs avec valeur par défaut). `PageImage.image` et `Book.output_pdf` restent en place. L'historique d'export peut référencer l'`output_pdf` existant.
- **Aperçu** : toute nouvelle étape du pipeline doit garder `preview_page == PDF`. Les tests existants (`test_process_image_gives_a4…`) le couvrent en partie et seront complétés.
- **Formulaire de réglages** : `preview_page` complète les champs absents avec les valeurs enregistrées. Les nouveaux champs doivent respecter ce mécanisme.
- **Passage en asynchrone** : la vue `process_book` change de comportement (redirection immédiate, puis suivi de l'état). Le test de bout en bout doit être adapté.

## 10. Décisions à prendre avant l'implémentation

1. **Secret exposé** : retirer `.env ` du suivi Git maintenant ? Réécrire l'historique ?
2. **File de tâches** : (a) worker léger maison (modèle `ProcessingJob` en base et commande `manage.py run_worker`, sans Redis, un service de plus dans Compose) ; (b) Celery avec Redis. Recommandation : **(a)**. Le volume (quelques livres à la fois) ne justifie pas Redis, et le passage à Celery restera simple si le modèle de job est bien découpé.
3. **Sidebar** : elle n'existe pas. La cible de la phase 2 liste environ 35 entrées, dont la plupart correspondent à des écrans inexistants. Proposition : créer la navigation latérale **avec les seules entrées réelles** (Tableau de bord, Mes livres, Nouveau livre, puis, dans un livre : Pages, Réglages, Export), et ajouter chaque rubrique quand son écran est livré.
4. **Mode B (PDF reconstruit)** : valider ReportLab et le périmètre « mise en page à blocs positionnés, avec signalement des pages complexes », sans recomposition fidèle garantie.
5. **Import PDF** : accepter des PDF scannés en entrée (rasterisation par page) ? Cela ajoute Ghostscript ou pdftoppm comme dépendance d'exécution, déjà présents dans l'image Docker.
6. **Polices** : uniquement des polices libres intégrées au dépôt (par ex. DejaVu, Liberation, Noto), ou aussi un téléversement par l'utilisateur ?

## 11. Plan d'implémentation proposé (une étape validée avant la suivante)

| Étape | Contenu | Fichiers principaux |
|---|---|---|
| 0 | Sécurité : `.env `, `.gitignore`, doublons dans les settings, HSTS facultatif, limite en pixels à l'import | `.gitignore`, `config/settings.py`, `books/views.py` |
| 1 | Extraire le JS inline vers `static/books/js/` et ajouter une navigation latérale (réelle) dans `base.html`, responsive, accessible au clavier | `templates/base.html`, `app.css`, `books/static/books/js/*.js` |
| 2 | Gestion des pages : suppression d'une page, réordonnancement (glisser-déposer et boutons), nettoyage des fichiers (signal `post_delete`), détection des doublons par hash | `models.py` (+ `sha256`), `views.py`, `urls.py`, migration |
| 3 | Tâches asynchrones : `ProcessingJob` (type, statut, progression par page, journal, erreur), worker, verrou par livre, reprise d'un livre bloqué en `processing`, polling `fetch` de l'état | `models.py`, `books/jobs.py`, `management/commands/run_worker.py`, `docker-compose.yml` |
| 4 | Recadrage par page : `PageImage.crop_quad` (JSON en coordonnées source), éditeur à 4 poignées, conversion entre coordonnées affichées et coordonnées source, annuler/rétablir, application à plusieurs pages, validation serveur | `models.py`, `image_processing.py`, nouvelle vue et template, JS |
| 5 | OCR structuré : `OcrBlock` (bbox, ordre, texte reconnu, texte corrigé, confiance, langue), extraction hOCR, écran de correction ; une relance ne remplace jamais un `corrected_text` | `models.py`, `books/ocr.py`, vues et templates |
| 6 | Profils couleur (original, restauration douce, texte noir net, livre ancien, illustrations, N&B) avec intensité et neutralisation sélective des dominantes sur les zones de texte sombre (Lab + masque de texte) | `image_processing.py` (nouveau module `color.py`), tests sur des images synthétiques |
| 7 | Détection des zones (texte, figure, tableau, légende) par page, `PageElement` éditable, puis images sources et occurrences (`SourceImage`, `ImageOccurrence`) | `models.py`, `classify.py`, éditeur |
| 8 | Mode A : existant, plus un historique `PdfExport` (mode, paramètres figés, date, fichier) | `models.py`, `ocr.py` |
| 9 | Mode B : ReportLab, polices intégrées, styles global et par bloc, images placées, détection des débordements | `books/rebuild.py`, `fonts/` |
| 10 | Stabilisation, documentation (README) et bilan | — |

Chaque étape comprend ses tests (backend et traitement d'images synthétiques) et la mise à jour du README.
