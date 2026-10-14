"""Website dataset loading, validation, and serialization."""

import csv
import json
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from urllib.parse import urlsplit


def hostname(url: str) -> str:
    parsed = urlsplit(url if "://" in url else "https://" + url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"Expected an HTTP(S) website URL: {url!r}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URLs must not contain embedded credentials")
    return parsed.hostname.lower().rstrip(".").removeprefix("www.")


@dataclass(frozen=True)
class Website:
    url: str
    html: str
    label: int

    @cached_property
    def description(self):
        from fake_news.features import description_from_html

        return description_from_html(self.html)

    def __post_init__(self):
        if not isinstance(self.url, str) or not self.url.strip():
            raise ValueError("Each record needs a nonempty URL")
        hostname(self.url)
        if not isinstance(self.html, str):
            raise ValueError("HTML must be a string")
        if type(self.label) is not int or self.label not in (0, 1):
            raise ValueError("Labels must be integers: 0=real, 1=fake")


def load_records(path) -> list[Website]:
    path = Path(path)
    rows = []
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("label") not in {"0", "1"}:
                    raise ValueError("CSV label must be 0 or 1")
                rows.append(
                    {"url": row.get("url"), "html": row.get("html"), "label": int(row["label"])}
                )
    elif path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        raise ValueError("Dataset must be .jsonl or .csv")
    records = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != {"url", "html", "label"}:
            raise ValueError(f"Record {index} must contain exactly url, html, and label")
        records.append(Website(**row))
    if not records:
        raise ValueError(f"No records in {path}")
    return records


def write_records(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def validate_splits(train, validation):
    if set(r.label for r in train) != {0, 1}:
        raise ValueError("Training data must contain both labels")
    if set(r.label for r in validation) != {0, 1}:
        raise ValueError("Validation data must contain both labels")
    overlap = {hostname(r.url) for r in train} & {hostname(r.url) for r in validation}
    if overlap:
        raise ValueError(f"Training/validation hostname overlap: {sorted(overlap)[:5]}")
