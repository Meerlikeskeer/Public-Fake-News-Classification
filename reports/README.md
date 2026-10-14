# Website classifier validation

Measured on the original 2,002-site training split and 309-site validation split. No held-out test evaluation was performed.

| Features | Validation accuracy | Fake precision | Fake recall | Fake F1 |
| --- | ---: | ---: | ---: | ---: |
| Domain | 55.34% | 52.73% | 20.57% | 29.59% |
| Domain + keywords | 72.82% | 65.24% | 86.52% | 74.39% |
| Bag-of-words | 75.40% | 66.67% | 92.20% | 77.38% |
| Keywords + bag-of-words | 80.58% | 73.41% | 90.07% | 80.89% |
| GloVe | 78.64% | 74.51% | 80.85% | 77.55% |
| Keywords + bag-of-words + GloVe | 82.52% | 75.14% | 92.20% | 82.80% |

See [full report](validation.json) for train metrics, matrices, data hashes, feature counts, and versions. Seed: 42; bag-of-words cap: 300; embedding source: Stanford GloVe 6B 300d. No hyperparameter search was run for this report.

Reproduce from the repository root:

```bash
python scripts/benchmark.py --data data/websites --embeddings data/glove.6B.300d.txt \
  --output artifacts/benchmark.json
```

Obtain the GloVe file separately using the README instructions. These historical source labels and validation measurements do not establish current website reliability or article-level truth.
