# Data preparation

Builds the processed parquet files used by the experiments from two datasets:

- **PTC** — the Propaganda Techniques Corpus (SemEval-2020 Task 11).
- **ProText** — the custom dataset used in this thesis.

There is one notebook per task — they are identical except for whether the
`techniques` column is kept:

| Notebook | Output | `techniques` column |
|----------|--------|---------------------|
| [`build_span_dataset.ipynb`](build_span_dataset.ipynb) | data for **Span Identification** | dropped (only span boundaries are needed) |
| [`build_technique_dataset.ipynb`](build_technique_dataset.ipynb) | data for **Technique Classification** | kept (the multi-label targets) |

## Set the paths first

Both notebooks read their paths from the repo-root [`config.py`](../config.py).
Open it and fill in:

- `PTC_ARTICLES_FOLDER` — PTC `train-articles` folder
- `PTC_LABEL_FILE` — PTC `train-task2-TC.labels` file
- `PROTEXT_XLSX` — the ProText `.xlsx` file
- `SPAN_DATA_DIR` — where `build_span_dataset.ipynb` writes its parquets
- `TECHNIQUE_DATA_DIR` — where `build_technique_dataset.ipynb` writes its parquets

Run the notebooks from this `data_preparation/` directory so the
`import config` cell can find the repo root.

## Run

Open the notebook for the task you need, set the paths above, and run all cells.
The pipeline:

1. **`build_ptc_dataset`** — joins each PTC article with its technique-labelled
   spans into one row per (article, span, technique).
2. **`collapse_to_multilabel_spans`** — groups identical spans so each
   (article_id, span_start, span_end) holds a *list* of techniques (multi-label).
3. **ProText harmonisation** — maps ProText technique names to the 14 canonical
   PTC labels and aligns its columns (`Context → article_text`, `Span → span_text`).
4. **Merge** PTC + ProText into one dataframe.
5. **`split_by_article_id`** — 80/20 train/val split *by article* (so no article
   leaks across splits).
6. Writes `train.parquet` and `val.parquet` into the configured output dir
   (`SPAN_DATA_DIR` / `TECHNIQUE_DATA_DIR`). The span notebook drops
   `techniques`; the technique notebook keeps it.

## Point the experiments at the parquets

Nothing to do — the training scripts read `SPAN_TRAIN_PARQUET` /
`TECHNIQUE_TRAIN_PARQUET` (etc.) from the same `config.py`, derived from the two
output dirs above. (The `Evaluation.ipynb` notebooks still use their own path
constants — set those when you evaluate.)

> Adjust the train/val fraction or random seed in the `split_by_article_id`
> call if you need a different split.
