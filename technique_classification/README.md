# Task 2 — Technique Classification (TC)

Given a propagandistic span, predict which of the **14 propaganda techniques**
it uses (SemEval-2020 Task 11, sub-task TC). This is **multi-label** — a span
may carry more than one technique. Scoring uses the official micro-F1 in
`task-TC_scorer.py` (the SemEval scorer library lives in each experiment's
`src/`).

The 14 techniques are listed in
`experiment_*/propaganda-techniques-names-semeval2020task11.txt`.

## Common architecture

- **Encoder:** RoBERTa-large
- **Input:** a context window (~100 chars) around the span, with the span itself
  wrapped in `<SPAN> … <SPAN>` marker tokens
- **Head:** linear classifier over pooled representations → 14 sigmoid outputs
- **Loss:** `BCEWithLogitsLoss`, decision threshold 0.5
- The two tasks are independent — TC is trained and evaluated on **gold** spans.

## The experiments

| # | Folder | Pooling / loss |
|---|--------|----------------|
| 1 | `experiment_1` | `[CLS]` pooling, plain `BCEWithLogitsLoss` |
| 2 | `experiment_2` | **final TC model** — `[CLS]` + span-mean + span-max pooling, `BCEWithLogitsLoss(pos_weight=…)` to counter label imbalance |
| 3 | `experiment_3_llm` | GPT few-shot baseline (14-shot, no fine-tuning) |

## Train

```bash
cd experiment_2          # or experiment_1
python model.py          # reads train/val parquet, saves best_technique_model.pt
```

Edit the `pd.read_parquet(...)` paths and the config block at the top of
`model.py` first.

## Evaluate

Open the experiment's `Evaluation.ipynb`, point the checkpoint path at
`trained_models_archive/technique_classification/experiment_<n>.pt`, and run.
It produces a SemEval submission `.tsv` and scores it with `task-TC_scorer.py`
(using the `src/` scorer library) against the gold labels.

## LLM baseline (Experiment 3)

```bash
export OPENAI_API_KEY=...
cd experiment_3_llm
python llm_technique.py --help   # run: 14-shot prompting -> output/results.csv
```

- **Run:** `llm_technique.py` — one in-context example per technique; writes
  predictions to `output/results.csv`.
- **Evaluate:** `Evaluation.ipynb` — scores `output/results.csv` with the same
  span-level micro-F1 as `task-TC_scorer.py`, plus per-technique F1 and error
  analysis.
