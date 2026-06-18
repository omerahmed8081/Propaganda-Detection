"""
evaluate_llm.py
Evaluates LLM technique-classification results using the same span-level
scoring logic as task-TC_scorer.py (SemEval 2020 Task 11).

Reads:  output/results.csv
Writes: output/llm_gold.tsv
        output/llm_submission.tsv
        output/llm_metrics.json
        output/llm_per_technique_f1.csv

Run from LLM/:
    python evaluate_llm.py
"""

import ast
import json
import os
import sys
import pandas as pd

RESULTS_CSV = os.path.join(os.path.dirname(__file__), "output", "results.csv")
OUT_DIR     = os.path.join(os.path.dirname(__file__), "output")


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_list(s):
    try:
        v = ast.literal_eval(str(s))
        return v if isinstance(v, list) else []
    except Exception:
        return []


def score(gold_dict, pred_dict):
    all_keys = set(gold_dict) | set(pred_dict)
    tp = fp = fn = 0
    for key in all_keys:
        g = gold_dict.get(key, set())
        p = pred_dict.get(key, set())
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return prec, rec, f1, tp, fp, fn


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(RESULTS_CSV)
    print(f"Loaded {len(df)} rows from {RESULTS_CSV}")

    df["gold_list"] = df["gold_techniques"].apply(parse_list)
    df["pred_list"] = df["predicted_techniques"].apply(parse_list)

    # Build {(article_id, span_start, span_end): set of techniques}
    # results.csv carries article_id but NOT span_start/span_end.
    # Use a surrogate key: (article_id, idx) — unique per span.
    gold_dict = {}
    pred_dict = {}

    for _, row in df.iterrows():
        key = (str(row["article_id"]), int(row["idx"]))
        gold_dict[key] = set(row["gold_list"])
        pred_dict[key] = set(row["pred_list"])

    # ── Overall metrics ───────────────────────────────────────────────────────
    prec, rec, f1, tp, fp, fn = score(gold_dict, pred_dict)

    print("\n🚀 LLM EVALUATION RESULTS")
    print(f"  Spans evaluated : {len(df)}")
    print(f"  Exact match     : {df['exact_match'].mean():.4f}")
    print(f"  Precision       : {prec:.6f}")
    print(f"  Recall          : {rec:.6f}")
    print(f"  F1              : {f1:.6f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}")

    # ── Per-technique ─────────────────────────────────────────────────────────
    all_labels = sorted({t for s in gold_dict.values() for t in s} |
                        {t for s in pred_dict.values() for t in s})

    all_keys  = set(gold_dict) | set(pred_dict)
    per_class = {}

    print("\n📊 Per-technique F1:")
    rows = []
    for label in all_labels:
        tp_c = fp_c = fn_c = 0
        for key in all_keys:
            g = gold_dict.get(key, set())
            p = pred_dict.get(key, set())
            if label in g and label in p:   tp_c += 1
            elif label not in g and label in p: fp_c += 1
            elif label in g and label not in p: fn_c += 1

        p_c  = tp_c / (tp_c + fp_c + 1e-9)
        r_c  = tp_c / (tp_c + fn_c + 1e-9)
        f1_c = 2 * p_c * r_c / (p_c + r_c + 1e-9)

        gold_count = sum(1 for g in gold_dict.values() if label in g)
        pred_count = sum(1 for p in pred_dict.values() if label in p)

        per_class[label] = f1_c
        print(f"  {label:<45}  F1={f1_c:.4f}  (gold={gold_count}, pred={pred_count})")
        rows.append({"technique": label, "f1": f1_c,
                     "precision": p_c, "recall": r_c,
                     "gold_count": gold_count, "pred_count": pred_count,
                     "tp": tp_c, "fp": fp_c, "fn": fn_c})

    # ── Comparison table ─────────────────────────────────────────────────────
    exp2_f1 = {
        "Appeal_to_Authority":                  0.7273,
        "Appeal_to_fear-prejudice":             0.8435,
        "Bandwagon,Reductio_ad_hitlerum":       0.7333,
        "Black-and-White_Fallacy":              0.8421,
        "Causal_Oversimplification":            0.8824,
        "Doubt":                                0.8973,
        "Exaggeration,Minimisation":            0.8824,
        "Flag-Waving":                          0.8980,
        "Loaded_Language":                      0.9335,
        "Name_Calling,Labeling":                0.9211,
        "Repetition":                           0.8521,
        "Slogans":                              0.8283,
        "Thought-terminating_Cliches":          0.7692,
        "Whataboutism,Straw_Men,Red_Herring":   0.5974,
    }

    print("\n📈 LLM vs Experiment-2 (RoBERTa-large) — per-technique F1:")
    print(f"  {'Technique':<45}  {'LLM':>6}  {'Exp2':>6}  {'Δ':>7}")
    print("  " + "-" * 68)
    for t in all_labels:
        llm_f1  = per_class.get(t, 0.0)
        e2_f1   = exp2_f1.get(t, None)
        delta   = f"{llm_f1 - e2_f1:+.4f}" if e2_f1 is not None else "  N/A"
        e2_str  = f"{e2_f1:.4f}" if e2_f1 is not None else "  N/A"
        print(f"  {t:<45}  {llm_f1:6.4f}  {e2_str}  {delta}")

    exp2_overall = 0.8878
    llm_overall  = f1
    print(f"\n  {'OVERALL (micro F1)':<45}  {llm_overall:6.4f}  {exp2_overall:.4f}  {llm_overall-exp2_overall:+.4f}")
    print(f"\n  Note: LLM evaluated on {len(df)} train+val spans (test set not yet run).")
    print(f"  Exp2 evaluated on 1,661 test spans. Direct comparison is indicative only.")

    # ── Save outputs ──────────────────────────────────────────────────────────
    metrics = {
        "num_spans":   len(df),
        "exact_match": float(df["exact_match"].mean()),
        "precision":   prec,
        "recall":      rec,
        "micro_f1":    f1,
        "tp": tp, "fp": fp, "fn": fn,
        "per_technique": {r["technique"]: r["f1"] for r in rows},
        "note": "Evaluated on train+val spans (11001). Test set not yet run."
    }
    with open(os.path.join(OUT_DIR, "llm_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    pd.DataFrame(rows).sort_values("f1", ascending=False).to_csv(
        os.path.join(OUT_DIR, "llm_per_technique_f1.csv"), index=False
    )
    print(f"\n✅ Saved: output/llm_metrics.json  output/llm_per_technique_f1.csv")


if __name__ == "__main__":
    main()
