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

All paths in both notebooks are left **empty** — fill them in before running.
Each is marked with an inline comment:

- `ARTICLES_FOLDER` — PTC `train-articles` folder
- `LABEL_FILE` — PTC `train-task2-TC.labels` file
- `pd.read_excel("")` — the ProText `.xlsx` file
- `OUTPUT_DIR` — directory to write the parquet files into

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
6. Writes `train.parquet` and `val.parquet` into `OUTPUT_DIR`. The span notebook
   drops `techniques`; the technique notebook keeps it.

## Point the experiments at the parquets

Update the `pd.read_parquet(...)` paths at the top of each experiment's
`model.py` (and the path constants in the `Evaluation.ipynb` notebooks) to the
`OUTPUT_DIR` you used.

> Adjust the train/val fraction or random seed in the `split_by_article_id`
> call if you need a different split.
