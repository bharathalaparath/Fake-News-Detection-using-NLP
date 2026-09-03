# Fake News Detection (NLP)

A Flask web app that classifies submitted article text as **Real** or **Fake**
news, backed by a scikit-learn TF-IDF + classifier pipeline.

## Architecture

```
preprocessing.py       # shared text-cleaning function (training + serving)
training/train.py      # trains a TF-IDF+classifier Pipeline, saves model.pkl
scripts/prepare_data.py# builds data/processed/{train,test}.csv from raw Kaggle CSVs
app/                    # Flask app factory + routes
templates/index.html   # the single form/result page
wsgi.py                 # WSGI/dev-server entrypoint
tests/                  # pytest suite (uses a small fixture, not the full dataset)
reference/              # old scratch/demo code and retired datasets, kept for history only
```

The model is trained and served as a **single sklearn `Pipeline`**
(`TfidfVectorizer` + classifier), saved together with `joblib`. This matters:
an earlier version of this project pickled only the classifier and applied a
different text-cleaning function at serving time than at training time, which
meant predictions were unreliable. Keeping one shared `preprocessing.py` and
one saved pipeline eliminates that class of bug by construction — there's
nothing to keep in sync.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv\Scripts\activate.bat for cmd.exe
pip install -r requirements-dev.txt
```

## Getting training data

See [data/README.md](data/README.md) for how to download the Kaggle
"Fake and Real News Dataset" and prepare it:

```bash
# after placing Fake.csv / True.csv under data/raw/
python scripts/prepare_data.py
```

**Note the dataset's license is CC BY-NC-SA 4.0 (non-commercial).** Confirm
that's acceptable for your use case before treating the resulting model as a
production asset — see data/README.md for details.

## Training

```bash
python training/train.py
```

Trains Logistic Regression, Naive Bayes, Random Forest, and SVM candidates,
picks the best by test accuracy, and writes:
- `model.pkl` — the full pipeline (not committed to git; see `.gitignore`)
- `training/artifacts/confusion_matrix_<model>.png`

## Running the app locally

```bash
python wsgi.py
```

Visit http://127.0.0.1:5000. If no `model.pkl` is present yet, the app still
starts; `/health` reports `"degraded"` and the form shows an error instead of
crashing until a model is trained and available at `MODEL_PATH` (default
`model.pkl` in the working directory).

Environment variables:
| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `model.pkl` | Path to the trained pipeline to load |
| `HOST` | `127.0.0.1` | Bind host for the dev server |
| `PORT` | `5000` | Bind port for the dev server |
| `FLASK_DEBUG` | `false` | Enables the Flask/Werkzeug debugger — **never set true in production** |

## Tests

```bash
pytest -q
```

Tests use a small hand-written fixture (`tests/fixtures/sample_articles.csv`)
to train a throwaway pipeline in-memory, so they don't depend on the Kaggle
download. Covers: preprocessing behavior, the app's routes (happy path,
empty input, missing-model degradation), and a save/reload sanity check on
the pipeline architecture itself.

## Running via Docker

```bash
docker build -t fake-news-detector .
docker run --rm -p 8000:8000 -v "$(pwd)/model.pkl:/app/model.pkl:ro" fake-news-detector
```

or `docker compose up --build` (see `docker-compose.yml`). The image runs via
`gunicorn`, not the Flask dev server. `model.pkl` is deliberately not baked
into the image (see the comment in `Dockerfile`) — mount it as a volume, or
add a build stage that fetches it from wherever you publish trained models
(GitHub Release asset, artifact registry, object storage). No specific cloud
target is assumed; this image runs on any container host.

## CI/CD

Two equivalent pipelines are provided (lint + test on every push/PR):
- `.github/workflows/ci.yml` — GitHub Actions
- `Jenkinsfile` — declarative Jenkins pipeline; requires a Jenkins agent with
  Python 3.11+ on `PATH`. Webhook/credential setup on the Jenkins server
  itself is outside this repo's scope.

Neither pipeline retrains the model automatically — that needs the
(gitignored) dataset and is intended to be a deliberate, separate step.

## Known open items

- **Dataset license**: confirm CC BY-NC-SA 4.0 is acceptable for your
  intended use before calling this "production."
- **Model artifact hosting**: decide between publishing trained pipelines as
  GitHub Release assets vs. building/mounting them at deploy time, once a
  deployment target is chosen.
