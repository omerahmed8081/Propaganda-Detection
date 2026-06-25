# Fine-Grained Propaganda Detection in News Articles

Code for a master's thesis on fine-grained propaganda detection, following the
**SemEval-2020 Task 11** formulation. The work covers two independent sub-tasks:

| Task | Name | Goal | Models |
|------|------|------|--------|
| **SI** | Span Identification | Find the character spans of propagandistic text | `span_identification/` — 7 supervised experiments + 1 LLM baseline |
| **TC** | Technique Classification | Given a span, predict which propaganda technique(s) it uses (14 labels, multi-label) | `technique_classification/` — 2 supervised experiments + 1 LLM baseline |

The two tasks are **independent**: the technique classifier is trained and
evaluated on *gold* spans, not on spans predicted by the SI model.

> **Trained model weights are not in this repository** 
---

## Repository structure

```
propaganda-detection/
├── README.md                      ← you are here
├── config.py                      ← central paths (raw data, processed data) — set these first
├── requirements.txt
├── data_preparation/
│   ├── README.md                       ← how to build the parquets
│   ├── build_span_dataset.ipynb        ← PTC + ProText → SI parquets (drops techniques)
│   └── build_technique_dataset.ipynb   ← PTC + ProText → TC parquets (keeps techniques)
│
├── span_identification/           ← Task 1 (SI)
│   ├── experiment_1_roberta/                   RoBERTa-large + BiLSTM + CRF (baseline)
│   ├── experiment_2_roberta_pos_ner/           + POS & NER embeddings
│   ├── experiment_3_roberta_pos_ner_discourse/ + 11-dim discourse features
│   ├── experiment_4_roberta_pos_ner_weighted/  + class-weighted loss
│   ├── experiment_5_pos_ner_discourse_weighted/ discourse + class weights
│   ├── experiment_6_limited_set/               reduced feature/label set
│   ├── experiment_7_updated_discourse/         FINAL SI model (best)
│   ├── experiment_8_llm/                        GPT 5.4 few-shot baseline
│   └── task-SI_scorer.py                         SI scorer
│
└── technique_classification/      ← Task 2 (TC)
    ├── experiment_1/              RoBERTa-large + span pooling + BCE
    ├── experiment_2/             FINAL TC model (best) — CLS + span-mean + span-max pooling
    ├── experiment_3_llm/         GPT 5.4 few-shot baseline
    └── (each supervised experiment ships the task-TC_scorer.py + src/)
```

---

## Setup

```bash
# Python 3.12 recommended
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

A CUDA GPU is strongly recommended — the experiments were run on an NVIDIA A100
(80 GB). RoBERTa-large + the 512-token sliding window fits comfortably in far
less, but training on CPU is impractical.

---

## Configuration

All data paths live in one place — [`config.py`](config.py) at the repo root.
Set them once and both the dataset builders and the model training scripts pick
them up:

| Setting | Used by |
|---------|---------|
| `PTC_ARTICLES_FOLDER`, `PTC_LABEL_FILE`, `PROTEXT_XLSX` | builders |
| `SPAN_DATA_DIR` | span builder → span `model.py` |
| `TECHNIQUE_DATA_DIR` | technique builder → technique `model.py` |
| `DEV_ARTICLES_FOLDER`, `DEV_SI_GOLD_FILE` | span `Evaluation.ipynb` |

`SPAN_TRAIN_PARQUET`, `TECHNIQUE_TRAIN_PARQUET`, `SI_SCORER`, etc. are derived
automatically (the technique notebooks evaluate on `TECHNIQUE_TEST_PARQUET`).
**Trained checkpoints are not configured here** — each experiment loads/saves
its best `.pt` in its own folder.

---

## 1. Build the dataset

The experiments use two datasets: the **PTC** corpus (SemEval-2020 Task 11) and
the custom **ProText** dataset. There is one builder per task (see
[`data_preparation/README.md`](data_preparation/README.md)):

- [`build_span_dataset.ipynb`](data_preparation/build_span_dataset.ipynb) — for
  Span Identification.
- [`build_technique_dataset.ipynb`](data_preparation/build_technique_dataset.ipynb)
  — for Technique Classification.

> **Paths:** set all data paths once in [`config.py`](config.py).

---

## 2. Train a model

Every supervised experiment trains the same way — run its `model.py`:

```bash

cd span_identification/experiment_7_updated_discourse
python model.py            


cd technique_classification/experiment_2
python model.py     
```

Hyper-parameters live in the `CFG` dataclass (SI) or the config block (TC) at
the top of each `model.py`.

## 3. Evaluate

Open the experiment's `Evaluation.ipynb`. It loads a trained `.pt`, predicts on
the PTC **dev** articles. Point the checkpoint path at the matching file in `trained_models_archive/` (see below).

## LLM baselines

Each LLM folder has a **run** script and an **evaluation** notebook:

- **SI — Experiment 8** (`span_identification/experiment_8_llm/`): `llm_span.py`
  (few-shot prompting → predicted span strings) + `Evaluation.ipynb` (offsets →
  scorer).
- **TC — Experiment 3** (`technique_classification/experiment_3_llm/`):
  `llm_technique.py` (14-shot prompting → predicted labels) + `Evaluation.ipynb`
  (span-level micro-F1).

The run scripts requries the API key to be changed (OPENAI_API_KEY) 


---

## Citation / data

- **PTC corpus** — Da San Martino et al., *SemEval-2020 Task 11: Detection of
  Propaganda Techniques in News Articles.*
- **ProText** — the custom dataset accompanying this thesis
  (https://github.com/Ahmadpir/ProText).

---

## Thesis

Published master's thesis: https://urn.fi/URN:NBN:fi-fe2026061268858
