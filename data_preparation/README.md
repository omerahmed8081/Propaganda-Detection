# Data preparation

Builds the processed parquet files used by every experiment from two raw
sources: the **PTC** corpus (SemEval-2020 Task 11) and the **ProText** dataset.

## 1. Obtain the raw data

### PTC (SemEval-2020 Task 11)
Register for / download the *Propaganda Techniques Corpus* from the task
organisers (https://propaganda.qcri.org/semeval2020-task11/). Arrange it as:

```
PTC/
├── train-articles/                 article<ID>.txt  (one news article per file)
├── dev-articles/                   article<ID>.txt
├── train-task2-TC.labels           article_id <tab> technique <tab> span_start <tab> span_end
├── train-task1-SI.labels           article_id <tab> span_start <tab> span_end
└── dev-task-SI.labels              gold spans for the dev set (used to score SI)
```

The dataset builder reads `train-articles/` + `train-task2-TC.labels` (the TC
labels carry both the span boundaries *and* the technique, so they serve both
tasks). The SI dev evaluation uses `dev-articles/` + `dev-task-SI.labels`.

### ProText
The custom thesis dataset, distributed at
https://github.com/Ahmadpir/ProText. Place `ProText.xlsx` so the notebook can
read it (columns: `Id`, `Context`, `Span`, `Techniques`, …).

## 2. Build the parquets

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

## 3. Point the experiments at the parquets

Update the `pd.read_parquet(...)` paths at the top of each experiment's
`model.py` (and the path constants in the `Evaluation.ipynb` notebooks) to your
local `processed_span_data/` location. The SI experiments use
`span_start`/`span_end`; the TC experiments use `techniques` + `span_text`.

> Adjust the train/val fraction or random seed in the `split_by_article_id`
> call if you need a different split.
