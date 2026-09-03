"""Canonical training pipeline for the fake news classifier.

Usage:
    python training/train.py [--data-dir data/processed] [--model-path model.pkl]

Expects data/processed/train.csv and data/processed/test.csv produced by
scripts/prepare_data.py, each with columns: text, label (label in {0, 1},
1 = fake). Trains several candidate models behind a single TF-IDF + classifier
Pipeline, picks the best by accuracy, and saves the *whole pipeline* (vectorizer
included) so app.py can call .predict() directly on raw text with no
train/serve mismatch.
"""
import argparse
import logging
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # noqa: E402 (must precede pyplot import; headless-safe backend)

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # noqa: E402
from sklearn.naive_bayes import MultinomialNB  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing import clean_text  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CANDIDATE_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42),
}


def load_split(data_dir: Path, split: str) -> pd.DataFrame:
    path = data_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/prepare_data.py first to generate "
            "data/processed/train.csv and data/processed/test.csv."
        )
    df = pd.read_csv(path)
    missing_cols = {"text", "label"} - set(df.columns)
    if missing_cols:
        raise ValueError(f"{path} is missing required column(s): {missing_cols}")
    return df


def train(data_dir: Path, model_path: Path, artifacts_dir: Path) -> None:
    log.info("Loading train/test splits from %s", data_dir)
    train_df = load_split(data_dir, "train")
    test_df = load_split(data_dir, "test")

    log.info("Train size: %d, Test size: %d", len(train_df), len(test_df))
    log.info("Train class distribution:\n%s", train_df["label"].value_counts())

    log.info("Cleaning text...")
    X_train = train_df["text"].fillna("").apply(clean_text)
    X_test = test_df["text"].fillna("").apply(clean_text)
    y_train = train_df["label"]
    y_test = test_df["label"]

    results = {}
    for name, classifier in CANDIDATE_MODELS.items():
        log.info("Training %s...", name)
        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(max_features=10000, min_df=2, max_df=0.8, ngram_range=(1, 2)),
                ),
                ("clf", classifier),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {"pipeline": pipeline, "accuracy": accuracy, "predictions": y_pred}
        log.info("%s accuracy: %.4f", name, accuracy)

    best_name = max(results, key=lambda name: results[name]["accuracy"])
    best_pipeline = results[best_name]["pipeline"]
    best_predictions = results[best_name]["predictions"]
    log.info("Best model: %s (accuracy %.4f)", best_name, results[best_name]["accuracy"])

    log.info(
        "Classification report for %s:\n%s",
        best_name,
        classification_report(y_test, best_predictions, target_names=["Real News", "Fake News"]),
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_path)
    log.info("Saved best pipeline (vectorizer + classifier) to %s", model_path)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, best_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real News", "Fake News"],
        yticklabels=["Real News", "Fake News"],
    )
    plt.title(f"Confusion Matrix - {best_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    cm_path = artifacts_dir / f"confusion_matrix_{best_name.replace(' ', '_')}.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    log.info("Saved confusion matrix to %s", cm_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-path", type=Path, default=Path("model.pkl"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("training/artifacts"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data_dir, args.model_path, args.artifacts_dir)
