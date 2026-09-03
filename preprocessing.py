"""Shared text preprocessing for both training and serving.

Using the same function in training/train.py and app/ guarantees there is no
train/serve skew: whatever transformation the model was trained on is exactly
what gets applied to text submitted through the web app.
"""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_NLTK_RESOURCES = {
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}


def ensure_nltk_data() -> None:
    """Download required NLTK corpora if they aren't already present.

    Safe to call repeatedly: nltk.data.find() is a local lookup, so once the
    data is cached this is a no-op with no network access.
    """
    for package, resource_path in _NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package, quiet=True)


ensure_nltk_data()

_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Clean and normalize a raw article/headline string for the model.

    Steps: lowercase, strip URLs, strip non-letter characters and digits,
    collapse whitespace, remove English stopwords, lemmatize each word.
    """
    if text is None:
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [word for word in words if word not in _STOP_WORDS]
    words = [_LEMMATIZER.lemmatize(word) for word in words]

    return " ".join(words)
