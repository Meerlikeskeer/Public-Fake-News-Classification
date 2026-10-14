"""Exploratory class fractions for domain and keyword hypotheses."""

import math

from fake_news.data import hostname


def hypothesis_fractions(records, hypothesis):
    fractions = []
    for label in (0, 1):
        subset = [r for r in records if r.label == label]
        fractions.append(sum(bool(hypothesis(r)) for r in subset) / len(subset) if subset else None)
    real, fake = fractions
    ratio = None
    if real is not None and fake is not None:
        ratio = fake / real if real else (math.inf if fake else None)
    return {
        "real_fraction": real,
        "fake_fraction": fake,
        "fake_to_real_ratio": "infinity" if ratio == math.inf else ratio,
    }


def analyze(records, extension=None, word=None, threshold=3):
    if extension is not None:
        if not extension.startswith("."):
            raise ValueError("Domain extension must start with a dot")
        return hypothesis_fractions(records, lambda r: hostname(r.url).endswith(extension.lower()))
    if word is not None and word.strip():
        return hypothesis_fractions(
            records, lambda r: r.html.lower().count(word.lower()) > threshold
        )
    raise ValueError("Choose a nonempty --word or --extension")
