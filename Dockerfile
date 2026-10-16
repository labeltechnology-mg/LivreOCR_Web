# syntax=docker/dockerfile:1
FROM python:3.15-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OCR engine and the tools ocrmypdf calls. Languages match the choices offered in the app.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl ghostscript qpdf unpaper \
        tesseract-ocr tesseract-ocr-osd tesseract-ocr-eng tesseract-ocr-fra tesseract-ocr-deu \
        tesseract-ocr-lat tesseract-ocr-spa tesseract-ocr-por \
    # German Fraktur ("deu_latf") is not packaged by Debian: add it next to the other models.
    && tessdir="$(dirname "$(find /usr/share/tesseract-ocr -name eng.traineddata | head -n 1)")" \
    && curl -fsSL -o "$tessdir/deu_latf.traineddata" \
        https://github.com/tesseract-ocr/tessdata_fast/raw/main/deu_latf.traineddata \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY requirements-paddle.txt .
RUN pip install -r requirements-paddle.txt

COPY --chown=app:app . .

# Uploaded scans and generated PDFs live here: mount a volume on it to keep them.
ENV LIVREOCR_MEDIA_ROOT=/data/media
RUN mkdir -p /data/media && chown -R app:app /data

USER app

# Static files are collected at build time; the key below is a throwaway used only for this step.
RUN DJANGO_SECRET_KEY=build-only DJANGO_DEBUG=1 python manage.py collectstatic --noinput

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import socket; socket.create_connection(('127.0.0.1', 8000), 3).close()"

# OCR runs in a background thread; this timeout applies to HTTP requests, not job duration.
CMD ["sh", "-c", "python manage.py migrate --noinput && exec gunicorn config.wsgi:application --bind 0.0.0.0:8000 --workers ${WEB_CONCURRENCY:-2} --timeout ${GUNICORN_TIMEOUT:-900}"]
