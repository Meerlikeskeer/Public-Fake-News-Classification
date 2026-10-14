"""One fitted artifact for training, evaluation, and prediction."""

import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from fake_news.features import WebsiteFeatures


def metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average="binary", pos_label=1, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_fake": float(precision),
        "recall_fake": float(recall),
        "f1_fake": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "records": len(y_true),
    }


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Classifier:
    def __init__(
        self, mode="combined", max_features=300, embedding_path=None, dimension=300, seed=42
    ):
        self.features = WebsiteFeatures(mode, max_features, embedding_path, dimension)
        self.seed = seed

    def fit(self, records):
        self.features.fit(records)
        self.estimator = LogisticRegression(
            solver="liblinear", max_iter=2000, random_state=self.seed
        )
        self.estimator.fit(self.features.transform(records), [r.label for r in records])
        self.metadata = {
            "format_version": 1,
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "mode": self.features.mode,
            "seed": self.seed,
            "embeddings": bool(self.features.vectors_),
            "labels": {"0": "real", "1": "fake"},
        }
        return self

    def evaluate(self, records):
        predictions = self.estimator.predict(self.features.transform(records))
        return metrics([r.label for r in records], predictions)

    def predict(self, records):
        matrix = self.features.transform(records)
        labels = self.estimator.predict(matrix)
        fake_index = list(self.estimator.classes_).index(1)
        probabilities = self.estimator.predict_proba(matrix)[:, fake_index]
        return [
            {
                "url": record.url,
                "label": int(label),
                "prediction": "fake" if label else "real",
                "fake_probability": float(prob),
            }
            for record, label, prob in zip(records, labels, probabilities)
        ]

    def coefficients(self):
        return sorted(
            zip(self.features.names(), self.estimator.coef_[0].tolist()),
            key=lambda item: abs(item[1]),
            reverse=True,
        )

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        path.with_suffix(".metadata.json").write_text(json.dumps(self.metadata, indent=2) + "\n")

    @staticmethod
    def load(path):
        # joblib uses pickle: load artifacts produced by this project that you trust.
        model = joblib.load(path)
        if not isinstance(model, Classifier):
            raise ValueError("Not a fake-news classifier artifact")
        if model.metadata.get("sklearn") != sklearn.__version__:
            raise ValueError("Use the scikit-learn version recorded in model metadata")
        return model
