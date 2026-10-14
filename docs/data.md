# Dataset format

Use local JSONL or CSV files with three fields:

```json
{"url": "example.com", "html": "<meta name=\"description\" content=\"Example\">", "label": 0}
```

- `url`: HTTP(S) website URL or hostname.
- `html`: saved HTML as a string.
- `label`: integer `0` for real or `1` for fake.

CSV files need a header `url,html,label`. Quote HTML containing commas or newlines. Both training and validation need examples from both classes. Split by website before training: normalized hostname overlap is rejected. Related domains and syndicated content require additional grouping when preparing data.

Place files at `data/websites/train.jsonl` and `data/websites/validation.jsonl`, or pass other paths to the CLI. There is no automatic dataset download or pickle import. Preserve the source, collection details, annotation method, and applicable terms alongside externally obtained data.

The measured benchmark used 2,002 training records (1,046 real, 956 fake) and 309 validation records (168 real, 141 fake). The report identifies the exact files by SHA-256. Those external dataset files are not distributed in the repository; synthetic examples are included for offline installation checks.

Optional vectors use the plain-text GloVe format: one word followed by its coordinates on each line. Obtain pretrained vectors from their original provider and match `--dimension` to the file. Text-generation corpora are local UTF-8 files.
