"""Domain, keyword, bag-of-words, and averaged GloVe features."""

import math
import re
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup, SoupStrainer
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from fake_news.data import hostname

EXTENSIONS = (
    ".com",
    ".org",
    ".net",
    ".info",
    ".biz",
    ".ru",
    ".co.uk",
    ".co",
    ".tv",
    ".news",
    ".gov",
    ".edu",
    ".us",
)
KEYWORDS = ("scandal", "cover-up", "exposed", "trump", "biden", "clinton", "sports", "finance")
MODES = ("domain", "keyword", "bow", "glove", "combined")


def description_from_html(html):
    soup = BeautifulSoup(html, "lxml", parse_only=SoupStrainer("meta"))
    for attrs in (
        {"property": "og:description"},
        {"name": "og:description"},
        {"name": "description"},
        {"property": "description"},
    ):
        for tag in soup.find_all("meta", attrs=attrs):
            content = tag.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def domain_features(url):
    host = hostname(url)
    return [float(host.endswith(extension)) for extension in EXTENSIONS]


def keyword_features(url, html):
    html = html.lower()
    return domain_features(url) + [math.log1p(html.count(word)) for word in KEYWORDS]


def tokens(text):
    return re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*", text.lower())


def load_embeddings(path, vocabulary, dimension):
    vectors = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            word, _, values = line.partition(" ")
            if word not in vocabulary:
                continue
            vector = np.fromstring(values, sep=" ", dtype=np.float32)
            if vector.shape != (dimension,) or not np.isfinite(vector).all():
                raise ValueError(f"Invalid {dimension}-dimensional embedding for {word!r}")
            vectors[word] = vector
    if not vectors:
        raise ValueError("No training-description tokens matched the embedding file")
    return vectors


class WebsiteFeatures:
    """Fit vocabulary on training data only, and retain it with the classifier."""

    def __init__(self, mode="combined", max_features=300, embedding_path=None, dimension=300):
        if mode not in MODES:
            raise ValueError(f"Unknown feature mode: {mode}")
        if max_features <= 0 or dimension <= 0:
            raise ValueError("Feature count and embedding dimension must be positive")
        self.mode = mode
        self.max_features = max_features
        self.embedding_path = str(embedding_path) if embedding_path else None
        self.dimension = dimension

    def fit(self, records):
        descriptions = (
            [r.description for r in records] if self.mode in {"bow", "glove", "combined"} else []
        )
        self.vectorizer_ = None
        self.vectors_ = {}
        if self.mode in {"bow", "combined"}:
            self.vectorizer_ = CountVectorizer(max_features=self.max_features)
            try:
                self.vectorizer_.fit(descriptions)
            except ValueError as error:
                if "empty vocabulary" not in str(error):
                    raise
                self.vectorizer_ = None
        if self.mode == "glove" and not self.embedding_path:
            raise ValueError("GloVe mode requires --embeddings pointing to a text vector file")
        if self.embedding_path and self.mode in {"glove", "combined"}:
            vocabulary = {word for text in descriptions for word in tokens(text)}
            self.vectors_ = load_embeddings(self.embedding_path, vocabulary, self.dimension)
        self.fitted_ = True
        return self

    def transform(self, records):
        if not getattr(self, "fitted_", False):
            raise ValueError("Fit features before transforming records")
        blocks = []
        if self.mode == "domain":
            blocks.append(sparse.csr_matrix([domain_features(r.url) for r in records]))
        if self.mode in {"keyword", "combined"}:
            blocks.append(sparse.csr_matrix([keyword_features(r.url, r.html) for r in records]))
        descriptions = None
        if self.mode in {"bow", "glove", "combined"}:
            descriptions = [r.description for r in records]
        if self.mode in {"bow", "combined"}:
            blocks.append(
                self.vectorizer_.transform(descriptions)
                if self.vectorizer_ is not None
                else sparse.csr_matrix((len(records), 1))
            )
        if self.mode == "glove" or (self.mode == "combined" and self.embedding_path):
            values = np.zeros((len(records), self.dimension), dtype=np.float32)
            for i, text in enumerate(descriptions):
                found = [self.vectors_[word] for word in tokens(text) if word in self.vectors_]
                if found:
                    values[i] = np.mean(found, axis=0)
            blocks.append(sparse.csr_matrix(values))
        return sparse.hstack(blocks, format="csr")

    def names(self):
        names = []
        if self.mode in {"domain", "keyword", "combined"}:
            names += ["domain:" + extension for extension in EXTENSIONS]
        if self.mode in {"keyword", "combined"}:
            names += ["keyword:" + word for word in KEYWORDS]
        if self.mode in {"bow", "combined"}:
            names += (
                ["bow:" + word for word in self.vectorizer_.get_feature_names_out()]
                if self.vectorizer_ is not None
                else ["bow:empty"]
            )
        if self.mode == "glove" or (self.mode == "combined" and self.embedding_path):
            names += [f"glove:{i}" for i in range(self.dimension)]
        return names
