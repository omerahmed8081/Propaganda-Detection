# Data preparation

Builds the processed parquet files used by the experiments from two datasets:

- **PTC**.
- **ProText**.

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

## Point the experiments at the parquets

Nothing to do — the training scripts read `SPAN_TRAIN_PARQUET` /
`TECHNIQUE_TRAIN_PARQUET` (etc.) from the same `config.py`, derived from the two
output dirs above. (The `Evaluation.ipynb` notebooks still use their own path
constants — set those when you evaluate.)

> Adjust the train/val fraction or random seed in the `split_by_article_id`
> call if you need a different split.
