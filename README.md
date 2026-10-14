# Fake News Classifier

My Python implementation of a website-level fake news classifier. It combines domain features, HTML keyword counts, bag-of-words descriptions, and optional GloVe embeddings with logistic regression. It supports training, evaluation, saved-model prediction, and feature analysis.

**Project by Meer Mathur.** I am sharing this personal implementation to show how I applied what I learned through [Inspirit AI](https://www.inspiritai.com/), which I credit for the teaching and learning resources. The implementation has been consolidated and extended with AI-assisted development.

Their supplied notebooks, starter-code files, slides, course datasets, and other instructional resources are intentionally not attached or redistributed here because I have not confirmed permission to make those materials public. This repository is not an official or endorsed release. External datasets, pretrained embeddings, and dependencies remain their creators' work; the included example datasets are synthetic fixtures.

## Setup and quick start

Use Python 3.11 or 3.12 from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
fake-news train --train examples/train.jsonl --validation examples/validation.jsonl \
  --mode combined --model models/demo.joblib --report artifacts/demo.json
fake-news predict --model models/demo.joblib --url example.com --html examples/website.html
pytest
```

The examples are synthetic installation fixtures. Their scores do not measure real-world performance. Imports do not download data, train models, or fetch websites.

## Train with a dataset

Provide local JSONL or CSV files containing `url`, `html`, and `label`. Labels are `0 = real`, `1 = fake`. See [dataset format](docs/data.md).

```bash
fake-news train --train data/websites/train.jsonl \
  --validation data/websites/validation.jsonl --mode combined \
  --model models/classifier.joblib --report artifacts/report.json
fake-news evaluate --model models/classifier.joblib --data data/websites/validation.jsonl
fake-news predict --model models/classifier.joblib --url https://example.com
```

Training rejects overlapping normalized hostnames between the training and validation splits. The labels classify websites; the model does not verify individual articles or claims.

## Feature modes

| Mode | Features | Embeddings required? |
| --- | --- | --- |
| `domain` | 13 hostname suffix indicators | No |
| `keyword` | Domain features and 8 log HTML substring counts | No |
| `bow` | Up to 300 learned meta-description word counts | No |
| `glove` | Averaged word vectors from descriptions | Yes |
| `combined` | Keyword + bag-of-words, plus GloVe when supplied | Optional |

Obtain pretrained vectors from the [Stanford GloVe project](https://nlp.stanford.edu/projects/glove/) and extract `glove.6B.300d.txt` locally:

```bash
fake-news train --train data/websites/train.jsonl \
  --validation data/websites/validation.jsonl --mode combined \
  --embeddings data/glove.6B.300d.txt --dimension 300 \
  --model models/combined-glove.joblib --report artifacts/combined-glove.json
python scripts/benchmark.py --data data/websites \
  --embeddings data/glove.6B.300d.txt --output artifacts/benchmark.json
```

The bag-of-words vocabulary is fitted only on training descriptions. Saved models retain vectors for training tokens, so prediction does not need the original vector file. Missing descriptions and unknown tokens produce zero text features. Without `--embeddings`, `combined` uses keyword and bag-of-words features.

## Results

On a fixed dataset of 2,002 training websites and 309 validation websites, the full combined model achieved **82.52% validation accuracy** and **82.80% fake-class F1**. No held-out test accuracy is claimed.

See the [model comparison](reports/README.md), [full report](reports/validation.json), and [tested versions](reports/tested-environment.json). Reports include split hashes, seeds, versions, feature counts, and confusion matrices.

## Analysis and text generation

```bash
fake-news analyze --data data/websites/train.jsonl --extension .news
fake-news analyze --data data/websites/train.jsonl --word scandal --threshold 3
```

An optional, separate character-level LSTM supports text generation experiments:

```bash
pip install -e '.[generation]'
fake-news-generate train --corpus examples/corpus.txt --epochs 1 \
  --max-characters 2000 --output models/generator.keras
fake-news-generate sample --model models/generator.keras --seed-text 'the robot' --length 100
```

It uses a 40-character context, 100 LSTM units, a softmax output, and temperature sampling. Sampled text is marked synthetic.

## Repository structure

```text
src/fake_news/   Dataset validation, features, models, CLI, and optional LSTM
examples/       Synthetic datasets, HTML, and text corpus
scripts/        Feature-family benchmark
reports/        Validation results and tested environment
tests/          Feature, split, serialization, and generation checks
docs/data.md    Dataset schema
```

Run `ruff check .`, `ruff format --check .`, and `pytest` to verify changes. Large external datasets, embeddings, and trained models stay outside version control. Load only trusted joblib files and use the scikit-learn version in their metadata. Loading an untrusted model can execute code.

Live fetching has timeouts and a 10 MiB response limit; blocked or JavaScript-rendered pages may not work. This is a local CLI, not a hosted service with protections against untrusted URLs or model uploads. Predictions reflect learned dataset patterns, not a calibrated truth score or a judgment about a publisher.
