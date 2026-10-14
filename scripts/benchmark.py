"""Evaluate each feature family on a fixed website train/validation split."""

import argparse
import json
from pathlib import Path

from fake_news.data import load_records, validate_splits
from fake_news.model import Classifier, file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/websites"))
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--dimension", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark.json"))
    args = parser.parse_args()
    train_path, val_path = args.data / "train.jsonl", args.data / "validation.jsonl"
    train, validation = load_records(train_path), load_records(val_path)
    validate_splits(train, validation)
    report = {
        "protocol": "Fixed website train/validation split; no held-out test evaluation",
        "train_sha256": file_sha256(train_path),
        "validation_sha256": file_sha256(val_path),
        "results": {},
    }
    if args.embeddings:
        report["embeddings_sha256"] = file_sha256(args.embeddings)
    for mode, embeddings in [
        ("domain", None),
        ("keyword", None),
        ("bow", None),
        ("combined", None),
        *(([("glove", args.embeddings), ("combined", args.embeddings)]) if args.embeddings else []),
    ]:
        name = mode + ("_glove" if mode == "combined" and embeddings else "")
        model = Classifier(mode, embedding_path=embeddings, dimension=args.dimension).fit(train)
        result = {
            "metadata": model.metadata,
            "features": len(model.features.names()),
            "train": model.evaluate(train),
            "validation": model.evaluate(validation),
        }
        model.save(Path("models") / f"{name}.joblib")
        report["results"][name] = result
        print(name, result["validation"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
