# Fine-Grained Propaganda Detection in News Articles

Code for a master's thesis on fine-grained propaganda detection, following the
**SemEval-2020 Task 11** formulation and additionally evaluating on the custom
**ProText** dataset. The work covers two independent sub-tasks:

| Task | Name | Goal | Models |
|------|------|------|--------|
| **SI** | Span Identification | Find the character spans of propagandistic text | `span_identification/` — 7 supervised experiments + 1 LLM baseline |
| **TC** | Technique Classification | Given a span, predict which propaganda technique(s) it uses (14 labels, multi-label) | `technique_classification/` — 2 supervised experiments + 1 LLM baseline |

The two tasks are **independent**: the technique classifier is trained and
evaluated on *gold* spans, not on spans predicted by the SI model.

> **Trained model weights are not in this repository** (each is ~1.4 GB). They
> are archived separately — see [Trained models](#trained-models).

---

## Repository structure

```
propaganda-detection/
├── README.md                      ← you are here
├── requirements.txt
├── data_preparation/
│   ├── README.md                  ← how to obtain raw data + build the parquets
│   └── build_dataset.ipynb        ← PTC + ProText → train/val/dev/test parquets
│
├── span_identification/           ← Task 1 (SI)
│   ├── README.md
│   ├── experiment_1_roberta/                   RoBERTa-large + BiLSTM + CRF (baseline)
│   ├── experiment_2_roberta_pos_ner/           + POS & NER embeddings
│   ├── experiment_3_roberta_pos_ner_discourse/ + 11-dim discourse features
│   ├── experiment_4_roberta_pos_ner_weighted/  + class-weighted loss
│   ├── experiment_5_pos_ner_discourse_weighted/ discourse + class weights
│   ├── experiment_6_limited_set/               reduced feature/label set
│   ├── experiment_7_updated_discourse/         FINAL SI model (best)
│   ├── experiment_8_llm/                        GPT few-shot baseline
│   └── task-SI_scorer.py                        official SemEval SI scorer
│
└── technique_classification/      ← Task 2 (TC)
    ├── README.md
    ├── experiment_1/              RoBERTa-large + span pooling + BCE
    ├── experiment_2/             FINAL TC model (best) — CLS + span-mean + span-max pooling
    ├── experiment_3_llm/         GPT / OpenRouter few-shot baseline
    └── (each supervised experiment ships the official task-TC_scorer.py + src/)
```

Each supervised experiment folder contains:
- `model.py` — model definition **and** the training entry point (`python model.py`)
- `Evaluation.ipynb` + `evaluation_helper*.py` — scoring against the official scorer

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

## 1. Build the dataset

Raw corpora are **not** redistributed here. See
[`data_preparation/README.md`](data_preparation/README.md) for how to obtain the
**PTC** corpus (SemEval-2020 Task 11) and the **ProText** dataset, then run
[`data_preparation/build_dataset.ipynb`](data_preparation/build_dataset.ipynb)
to produce the processed parquet files:

```
processed_span_data/
├── train.parquet   # columns: article_id, span_start, span_end, article_text, span_text, techniques
├── val.parquet
├── dev.parquet
└── test.parquet
```

The same parquets feed **both** tasks: SI uses `span_start`/`span_end`, TC uses
`techniques` together with the span text.

> **Paths:** the training/eval scripts currently use absolute paths
> (e.g. `/home/.../processed_span_data/train.parquet`). After building the data,
> update the `read_parquet(...)` paths near the top of each `model.py` /
> `main()` and the path constants in the `Evaluation.ipynb` notebooks to point
> at your local copies.

---

## 2. Train a model

Every supervised experiment trains the same way — run its `model.py`:

```bash
# Span identification, e.g. the final model
cd span_identification/experiment_7_updated_discourse
python model.py            # reads train/val parquet, saves best_*.pt by val F1

# Technique classification, e.g. the final model
cd technique_classification/experiment_2
python model.py            # saves best_technique_model.pt
```

Hyper-parameters live in the `CFG` dataclass (SI) or the config block (TC) at
the top of each `model.py`.

## 3. Evaluate

Open the experiment's `Evaluation.ipynb`. It loads a trained `.pt`, predicts on
the PTC **dev** articles, and scores with the official SemEval scorer
(`task-SI_scorer.py` for SI, `task-TC_scorer.py` for TC). Point the checkpoint
path at the matching file in `trained_models_archive/` (see below).

## LLM baselines

- **SI — Experiment 8** (`span_identification/experiment_8_llm/`): few-shot
  prompting (positive/negative article examples) → JSON array of span strings.
- **TC — Experiment 3** (`technique_classification/experiment_3_llm/`): 14-shot
  prompting (one example per technique) → JSON array of technique labels.

Both read the API key from an environment variable — **no keys are committed**:

```bash
export OPENAI_API_KEY=...        # llm_span.py, llm_technique.py (GPT)
export OPENROUTER_API_KEY=...    # llm_openrouter.py (open models)
```

---

## Trained models

The 10 trained checkpoints (~14 GB total) are stored **outside this repo** in
`trained_models_archive/`, renamed per experiment. See
`trained_models_archive/MODELS_INDEX.md` for the mapping from each file to its
experiment, encoder, and validation score. They are excluded from git via
`.gitignore`; to publish them, use Git LFS or an external store (e.g. Zenodo,
university storage).

---

## Citation / data

- **PTC corpus** — Da San Martino et al., *SemEval-2020 Task 11: Detection of
  Propaganda Techniques in News Articles.*
- **ProText** — the custom dataset accompanying this thesis
  (https://github.com/Ahmadpir/ProText).
