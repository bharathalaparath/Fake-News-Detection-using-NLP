from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app import create_app
from preprocessing import clean_text

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_articles.csv"


@pytest.fixture
def trained_model_path(tmp_path):
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

    model_path = tmp_path / "model.pkl"
    joblib.dump(pipeline, model_path)
    return model_path


@pytest.fixture
def client(trained_model_path):
    app = create_app(model_path=str(trained_model_path))
    app.testing = True
    return app.test_client()


@pytest.fixture
def client_without_model(tmp_path):
    app = create_app(model_path=str(tmp_path / "does-not-exist.pkl"))
    app.testing = True
    return app.test_client()


def test_get_index_returns_form(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"<form" in response.data


def test_health_ok_when_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["model_loaded"] is True


def test_health_degraded_when_model_missing(client_without_model):
    response = client_without_model.get("/health")
    assert response.status_code == 503
    assert response.get_json()["model_loaded"] is False


def test_post_empty_text_shows_error(client):
    response = client.post("/", data={"txt": "   "})
    assert response.status_code == 200
    assert b"enter some article text" in response.data.lower()


def test_post_fake_like_text_predicts_fake(client):
    response = client.post(
        "/",
        data={
            "txt": "SHOCKING secret miracle cure doctors hate this one weird trick that changes everything"
        },
    )
    assert response.status_code == 200
    assert b"Fake News" in response.data


def test_post_real_like_text_predicts_real(client):
    response = client.post(
        "/",
        data={"txt": "The city council approved a new transit budget after committee review"},
    )
    assert response.status_code == 200
    assert b"Real News" in response.data


def test_post_without_model_shows_error(client_without_model):
    response = client_without_model.post("/", data={"txt": "some article text here"})
    assert response.status_code == 200
    assert b"not available" in response.data.lower()
