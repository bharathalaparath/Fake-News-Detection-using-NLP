FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK corpora at build time so the container never needs
# network access at runtime just to preprocess text.
COPY preprocessing.py .
RUN python -c "from preprocessing import ensure_nltk_data; ensure_nltk_data()"

COPY app/ app/
COPY templates/ templates/
COPY wsgi.py .

# model.pkl is intentionally not committed to git (see .gitignore) or copied
# in here. Provide it at deploy time by either:
#   1. Mounting it as a volume: -v /path/to/model.pkl:/app/model.pkl
#   2. Adding a `COPY model.pkl .` line in a custom build stage that first
#      fetches it from wherever you publish trained models (e.g. a GitHub
#      Release asset, artifact registry, or object storage).
# The app starts and serves /health as "degraded" if no model is present,
# rather than crashing, so this is safe to leave unset in CI image builds.

ENV FLASK_DEBUG=false \
    HOST=0.0.0.0 \
    PORT=8000 \
    MODEL_PATH=/app/model.pkl

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "wsgi:app"]
