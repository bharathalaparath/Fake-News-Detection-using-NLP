# Training data

## Source

This project trains on the Kaggle **"Fake and Real News Dataset"**
(ISOT Fake News Dataset, originally produced by the University of Victoria's
Information Security and Object Technology research group):

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

It ships two files:
- `True.csv` — ~21,400 real articles (Reuters)
- `Fake.csv` — ~23,500 fake articles (various flagged fake-news sites)

Columns: `title`, `text`, `subject`, `date`.

## ⚠️ License caution — read before using in production

The dataset is distributed under **CC BY-NC-SA 4.0**, which is
**non-commercial only**. Before treating a model trained on this data as a
"production" asset (especially in any client-facing or commercial context),
confirm this use qualifies, or swap in a dataset with a license that permits
it. This is called out as an open item in the project plan and needs an
explicit decision, not an assumption.

## How to get the data

1. Create a free Kaggle account if you don't have one, and (if required)
   generate an API token from your Kaggle account settings.
2. Download the dataset manually from the URL above (or via the Kaggle CLI:
   `kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset`).
3. Unzip it and place `Fake.csv` and `True.csv` under `data/raw/` in this
   repo:
   ```
   data/raw/Fake.csv
   data/raw/True.csv
   ```
   (`data/raw/` is gitignored — these files are never committed.)
4. Run the prep script to build the train/test splits used by training:
   ```bash
   python scripts/prepare_data.py
   ```
   This writes `data/processed/train.csv` and `data/processed/test.csv`
   (also gitignored), each with columns `title`, `text`, `label`
   (`label`: 1 = fake, 0 = real).
5. Train the model:
   ```bash
   python training/train.py
   ```

## Legacy/demo data

The datasets originally in this repo (`Test Data/Dataset01.csv`,
`Dataset01.xlsx`, `english_fake_news_2212.csv`, `evaluation.csv`) have been
moved to `reference/legacy-data/` and are **not used** for training anymore:
- `Dataset01.csv` has a header/data delimiter mismatch (comma header, semicolon
  rows) that corrupts the `label` column when parsed.
- `english_fake_news_2212.csv` is synthetic/templated data (the same
  boilerplate body text repeats across unrelated headlines), which does not
  reflect real article patterns and was degrading model quality.
- `evaluation.csv` has the same delimiter issue as `Dataset01.csv` and was
  never wired into the training script.

They're kept only for historical reference, not as an active data source.

## Tests

A small hand-written fixture (a handful of clearly real/fake examples) lives
at `tests/fixtures/sample_articles.csv` and is used by the test suite instead
of the full dataset, so tests don't depend on the Kaggle download.
