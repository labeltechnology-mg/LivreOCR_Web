from pathlib import Path

CSS_FILE = Path(__file__).resolve().parent / "static" / "books" / "css" / "app.css"

def asset_version(request):
    """Changes whenever app.css changes, so browsers never keep serving a stale stylesheet."""
    try:
        return {"asset_version": int(CSS_FILE.stat().st_mtime)}
    except OSError:
        return {"asset_version": 0}
