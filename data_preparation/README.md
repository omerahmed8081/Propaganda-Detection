# Data preparation

Builds the processed parquet files used by every experiment from two datasets:

- **PTC** — the Propaganda Techniques Corpus (SemEval-2020 Task 11).
- **ProText** — the custom dataset used in this thesis.

## Build the parquets

Open and run [`build_dataset.ipynb`](build_dataset.ipynb). It performs:

1. **`build_ptc_dataset`** — joins each PTC article with its technique-labelled
   spans into one row per (article, span, technique).
2. **`collapse_to_multilabel_spans`** — groups identical spans so each
   (article_id, span_start, span_end) holds a *list* of techniques (multi-label).
3. **ProText harmonisation** — maps ProText technique names to the 14 canonical
   PTC labels and aligns its columns (`Context → article_text`, `Span → span_text`).
4. **Merge** PTC + ProText into one dataframe.
5. **`split_by_article_id`** — 80/20 train/val split *by article* (so no article
   leaks across splits), and writes:

```
processed_span_data/
├── train.parquet
├── val.parquet
├── dev.parquet
└── test.parquet
```

Each row has the columns:
`article_id, span_start, span_end, article_text, span_text, techniques`.

The same parquets feed both tasks: SI uses `span_start`/`span_end`, TC uses
`techniques` + `span_text`.

## Point the experiments at the parquets

Update the `pd.read_parquet(...)` paths at the top of each experiment's
`model.py` (and the path constants in the `Evaluation.ipynb` notebooks) to your
local `processed_span_data/` location.

> Adjust the train/val fraction or random seed in the `split_by_article_id`
> call if you need a different split.
