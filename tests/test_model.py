"""Sanity check for the train/serve architecture: a Pipeline (vectorizer +
classifier) trained on raw text can be saved, reloaded, and used to predict
directly on raw text — no separate vectorizer step required at inference
time. Uses the small hand-written fixture, not the full Kaggle dataset.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from preprocessing import clean_text

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_articles.csv"


def _train_tiny_pipeline() -> Pipeline:
    df = pd.read_csv(FIXTURE_PATH)
    X = df["text"].apply(clean_text)
    y = df["label"]

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def test_pipeline_predicts_on_raw_text_after_roundtrip(tmp_path):
    pipeline = _train_tiny_pipeline()

    model_path = tmp_path / "model.pkl"
    joblib.dump(pipeline, model_path)
    loaded = joblib.load(model_path)

    fake_example = clean_text(
        "SHOCKING secret miracle cure doctors hate this one weird trick"
    )
    real_example = clean_text(
        "The city council approved a new budget after committee review"
    )

    predictions = loaded.predict([fake_example, real_example])

    assert set(predictions).issubset({0, 1})
    assert predictions[0] == 1
    assert predictions[1] == 0
