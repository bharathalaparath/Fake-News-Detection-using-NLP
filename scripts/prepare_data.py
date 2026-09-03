"""Prepare the Kaggle ISOT "Fake and Real News" dataset for training.

Download Fake.csv and True.csv manually from Kaggle (see data/README.md for
the exact steps and dataset URL) and place them under data/raw/ before running
this script.

Usage:
    python scripts/prepare_data.py [--raw-dir data/raw] [--out-dir data/processed] [--test-size 0.2]

Produces data/processed/train.csv and data/processed/test.csv, each with
columns: title, text, label (label: 1 = fake, 0 = real). `text` is the
title concatenated with the article body, which is what the model is trained
and served on.
"""
import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REQUIRED_RAW_COLUMNS = {"title", "text"}


def load_raw(raw_dir: Path) -> pd.DataFrame:
    fake_path = raw_dir / "Fake.csv"
    true_path = raw_dir / "True.csv"
    for path in (fake_path, true_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Download the Kaggle ISOT dataset and place "
                "Fake.csv / True.csv under data/raw/ — see data/README.md."
            )

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    for name, df in (("Fake.csv", fake_df), ("True.csv", true_df)):
        missing = REQUIRED_RAW_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing required column(s): {missing}")

    fake_df = fake_df.copy()
    true_df = true_df.copy()
    fake_df["label"] = 1
    true_df["label"] = 0

    return pd.concat([fake_df, true_df], ignore_index=True)


def build_dataset(raw_dir: Path) -> pd.DataFrame:
    df = load_raw(raw_dir)

    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")

    df = df[(df["title"].str.strip() != "") | (df["text"].str.strip() != "")]
    df = df.drop_duplicates(subset=["title", "text"])

    df["text"] = (df["title"] + " " + df["text"]).str.strip()

    return df[["title", "text", "label"]].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    log.info("Loading raw data from %s", args.raw_dir)
    df = build_dataset(args.raw_dir)
    log.info("Combined dataset shape: %s", df.shape)
    log.info("Class distribution:\n%s", df["label"].value_counts())

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["label"],
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.csv"
    test_path = args.out_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    log.info("Wrote %d train rows to %s", len(train_df), train_path)
    log.info("Wrote %d test rows to %s", len(test_df), test_path)


if __name__ == "__main__":
    main()
