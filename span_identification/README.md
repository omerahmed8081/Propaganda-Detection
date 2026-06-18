# Task 1 — Span Identification (SI)

Identify the character spans of propagandistic text in news articles
(SemEval-2020 Task 11, sub-task SI). Scoring uses the official partial-overlap
precision/recall/F1 in [`task-SI_scorer.py`](task-SI_scorer.py).

## Common architecture

All supervised experiments share the same backbone, adding one component at a
time (this is the ablation progression of the thesis):

- **Encoder:** RoBERTa-large, sliding window of `max_length=512`, `stride=256`
- **Tagging scheme:** BIOES (`O, B-PROP, I-PROP, E-PROP, S-PROP`)
- **Head:** feature fusion → BiLSTM → linear → **CRF**
- **Optional features:** POS embeddings, NER embeddings, an 11-dim discourse
  feature vector (sentence/paragraph position, quotation, title-likeness,
  contrast/cause/emphasis/conclusion discourse markers)
- **Loss:** CRF negative log-likelihood, optionally combined with a
  class-weighted cross-entropy term

## The experiments

| # | Folder | What it adds |
|---|--------|--------------|
| 1 | `experiment_1_roberta` | RoBERTa-large + BiLSTM + CRF baseline (no extra features) |
| 2 | `experiment_2_roberta_pos_ner` | + POS and NER embeddings |
| 3 | `experiment_3_roberta_pos_ner_discourse` | + 11-dim discourse features |
| 4 | `experiment_4_roberta_pos_ner_weighted` | POS/NER + class-weighted loss |
| 5 | `experiment_5_pos_ner_discourse_weighted` | discourse features + class weights |
| 6 | `experiment_6_limited_set` | reduced feature / context configuration |
| 7 | `experiment_7_updated_discourse` | **final SI model** — refined discourse features + hybrid CRF/CE loss, discriminative learning rates |
| 8 | `experiment_8_llm` | GPT few-shot baseline (no fine-tuning) |

## Train

```bash
cd experiment_7_updated_discourse      # or any experiment_1..7
python model.py
```

`model.py` reads `train.parquet`/`val.parquet`, trains, and saves the best
checkpoint (by validation official-F1). Edit the `read_parquet(...)` paths and
hyper-parameters in the `CFG` dataclass / `main()` at the top of `model.py`
first. (Experiment 7 includes a small grid in `main()`'s `EXPERIMENTS` list.)

## Evaluate

Open the experiment's `Evaluation.ipynb`, set the checkpoint path to the
matching file in `trained_models_archive/span_identification/`, and run. It
predicts on `PTC/dev-articles`, writes a predictions `.labels` file, and calls
`task-SI_scorer.py` against `PTC/dev-task-SI.labels`.

For reference, the final model (Experiment 7) scores **F1 ≈ 0.47** on PTC dev
with the official scorer.

## LLM baseline (Experiment 8)

```bash
export OPENAI_API_KEY=...
cd experiment_8_llm
python llm_span.py --help     # run: few-shot prompting -> results.json (predicted span strings)
```

- **Run:** `llm_span.py` — few-shot prompting with positive/negative article
  examples; the model returns a JSON array of propagandistic span strings,
  saved to `results.json`.
- **Evaluate:** `Evaluation.ipynb` — locates each predicted/gold span string in
  the article text to get character offsets, then scores with the official
  `task-SI_scorer.py`.

`llm_sentence_analysis.py` is an optional, deeper sentence-level error analysis
(uses external `.labels` paths — adjust them before running).
