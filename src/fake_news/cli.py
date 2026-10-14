"""Command-line interface; imports have no network or training side effects."""

import argparse
import json
from pathlib import Path

import requests

from fake_news.analysis import analyze
from fake_news.data import (
    Website,
    hostname,
    load_records,
    validate_splits,
)
from fake_news.features import MODES
from fake_news.model import Classifier, file_sha256

MAX_RESPONSE_BYTES = 10 * 1024 * 1024


def fetch_html(url):
    hostname(url)
    url = url if "://" in url else "https://" + url
    with requests.get(url, timeout=(10, 30), stream=True) as response:
        response.raise_for_status()
        chunks, size = [], 0
        for chunk in response.iter_content(64 * 1024):
            size += len(chunk)
            if size > MAX_RESPONSE_BYTES:
                raise ValueError("Live page exceeds the 10 MiB response limit; use saved HTML")
            chunks.append(chunk)
        return b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")


def build_parser():
    parser = argparse.ArgumentParser(description="Website classification: 0=real, 1=fake")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--train", required=True, type=Path)
    train.add_argument("--validation", required=True, type=Path)
    train.add_argument("--mode", choices=MODES, default="combined")
    train.add_argument("--max-features", type=int, default=300)
    train.add_argument("--embeddings", type=Path)
    train.add_argument("--dimension", type=int, default=300)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--model", type=Path, default=Path("models/classifier.joblib"))
    train.add_argument("--report", type=Path, default=Path("artifacts/report.json"))
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--data", type=Path, required=True)
    predict = commands.add_parser("predict")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--url", required=True)
    predict.add_argument("--html", type=Path, help="Read saved HTML instead of fetching URL")
    exploration = commands.add_parser("analyze")
    exploration.add_argument("--data", type=Path, required=True)
    choice = exploration.add_mutually_exclusive_group(required=True)
    choice.add_argument("--extension")
    choice.add_argument("--word")
    exploration.add_argument("--threshold", type=int, default=3)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            train, validation = load_records(args.train), load_records(args.validation)
            validate_splits(train, validation)
            model = Classifier(
                args.mode, args.max_features, args.embeddings, args.dimension, args.seed
            ).fit(train)
            result = {
                "metadata": model.metadata,
                "train": model.evaluate(train),
                "validation": model.evaluate(validation),
                "feature_count": len(model.features.names()),
                "sources": {
                    "train_sha256": file_sha256(args.train),
                    "validation_sha256": file_sha256(args.validation),
                },
                "coefficients": model.coefficients(),
            }
            if args.embeddings:
                result["sources"]["embeddings_sha256"] = file_sha256(args.embeddings)
            model.save(args.model)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2) + "\n")
        elif args.command == "evaluate":
            result = Classifier.load(args.model).evaluate(load_records(args.data))
        elif args.command == "predict":
            hostname(args.url)
            if args.html:
                html = args.html.read_text(encoding="utf-8")
            else:
                html = fetch_html(args.url)
            result = Classifier.load(args.model).predict([Website(args.url, html, 0)])[0]
        else:
            result = analyze(load_records(args.data), args.extension, args.word, args.threshold)
    except (ValueError, OSError, requests.RequestException) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
