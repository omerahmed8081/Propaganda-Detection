"""
Central path configuration for the propaganda-detection repo.

Fill in the paths below once. The dataset builders
(`data_preparation/build_*_dataset.ipynb`) and the model training scripts
(`*/model.py`) import their paths from here, so you don't edit each file.

Note: trained checkpoints are NOT configured here — each experiment saves its
best `.pt` inside its own folder.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. RAW DATA  (inputs to the data_preparation notebooks)
# ---------------------------------------------------------------------------
PTC_ARTICLES_FOLDER = ""   # PTC train-articles folder
PTC_LABEL_FILE      = ""   # PTC train-task2-TC.labels file
PROTEXT_XLSX        = ""   # ProText .xlsx file

# ---------------------------------------------------------------------------
# 2. PROCESSED DATA  (output of the builders, input to the models)
#    Span and technique need different parquets — the span builder drops the
#    `techniques` column, the technique builder keeps it — so each writes to
#    its own directory.
# ---------------------------------------------------------------------------
SPAN_DATA_DIR      = ""   # build_span_dataset.ipynb writes train/val.parquet here
TECHNIQUE_DATA_DIR = ""   # build_technique_dataset.ipynb writes train/val.parquet here


# ---------------------------------------------------------------------------
# Derived parquet paths read by the model.py training scripts.
# These follow from the two directories above — no need to edit them.
# ---------------------------------------------------------------------------
def _p(directory, name):
    return str(Path(directory) / name) if directory else ""


SPAN_TRAIN_PARQUET = _p(SPAN_DATA_DIR, "train.parquet")
SPAN_VAL_PARQUET   = _p(SPAN_DATA_DIR, "val.parquet")

TECHNIQUE_TRAIN_PARQUET = _p(TECHNIQUE_DATA_DIR, "train.parquet")
TECHNIQUE_VAL_PARQUET   = _p(TECHNIQUE_DATA_DIR, "val.parquet")
TECHNIQUE_TEST_PARQUET  = _p(TECHNIQUE_DATA_DIR, "test.parquet")
