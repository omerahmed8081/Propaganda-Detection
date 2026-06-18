"""
LLM Sentence-Level Propaganda Analysis
=======================================
Standard scoring penalises the LLM for predicting full sentences instead of
exact sub-spans.  This script asks:

  "If we give the LLM credit whenever its prediction contains the sentence
   that holds a gold propaganda span, how well did it actually perform?"

Three evaluation modes are run:
  1. Official partial-overlap (the existing F1=0.389 — for reference)
  2. Sentence-level recall  : gold span is 'found' if LLM predicted any span
                               whose character range contains the gold span's
                               sentence
  3. Sentence-level F1      : precision / recall both measured at sentence
                               level (sentence is the unit, not the character)
"""

import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
GOLD_FILE  = "/home/omer_ahmed/Experiments/gold.labels"
LLM_FILE   = "/home/omer_ahmed/Experiments/llm_predictions.labels"
ART_DIR    = "/home/omer_ahmed/Experiments/PTC/dev-articles"
OUT_DIR    = "/home/omer_ahmed/Experiments/llm_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Design ────────────────────────────────────────────────────────────────────
BG     = "#0D1B2A"
PANEL  = "#112233"
ACCENT = "#00B4D8"
WHITE  = "#FFFFFF"
LGRAY  = "#B0BEC5"
MGRAY  = "#546E7A"
GREEN  = "#26A69A"
ORANGE = "#FF7043"
PURPLE = "#AB47BC"


# ────────────────────────────────────────────────────────────────────────────
# 1. LOADERS
# ────────────────────────────────────────────────────────────────────────────
def load_labels(path):
    spans = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            aid, s, e = parts[0], int(parts[1]), int(parts[2])
            spans[aid].append((s, e))
    return dict(spans)


def load_articles(folder):
    texts = {}
    for fn in os.listdir(folder):
        if fn.endswith(".txt"):
            aid = fn.replace("article", "").replace(".txt", "")
            with open(os.path.join(folder, fn), encoding="utf-8") as f:
                texts[aid] = f.read()
    return texts


# ────────────────────────────────────────────────────────────────────────────
# 2. SENTENCE SPLITTING
# ────────────────────────────────────────────────────────────────────────────
def split_sentences(text):
    """
    Returns list of (start, end) character pairs for each sentence.
    Strategy: split on newlines first, then split runs on sentence-ending
    punctuation (.!?) followed by whitespace+capital letter.
    """
    sentences = []
    # split on blank lines / newlines
    cursor = 0
    for block in re.split(r"\n+", text):
        if not block.strip():
            cursor += len(block) + 1
            continue
        block_start = text.find(block, cursor)
        # split block into sentences on .  !  ?  followed by space + capital
        sub_sents = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", block)
        sub_cursor = block_start
        for ss in sub_sents:
            ss_start = text.find(ss, sub_cursor)
            if ss_start == -1:
                sub_cursor += len(ss)
                continue
            ss_end = ss_start + len(ss)
            if ss_end > ss_start:
                sentences.append((ss_start, ss_end))
            sub_cursor = ss_end
        cursor = block_start + len(block) + 1

    return sentences


def find_sentence_for_span(span_start, span_end, sentences):
    """Return the sentence (start, end) that best contains this span."""
    best = None
    best_overlap = 0
    for s_start, s_end in sentences:
        overlap = max(0, min(span_end, s_end) - max(span_start, s_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best = (s_start, s_end)
    return best


# ────────────────────────────────────────────────────────────────────────────
# 3. OFFICIAL PARTIAL-OVERLAP SCORING (same as the existing scorer)
# ────────────────────────────────────────────────────────────────────────────
def partial_overlap_f1(gold_by_article, pred_by_article):
    prec_num = prec_den = 0.0
    rec_num  = rec_den  = 0.0

    all_aids = set(gold_by_article) | set(pred_by_article)
    for aid in all_aids:
        gold = gold_by_article.get(aid, [])
        pred = pred_by_article.get(aid, [])

        for ps, pe in pred:
            plen = pe - ps
            if plen == 0:
                continue
            best = max(
                (max(0, min(pe, ge) - max(ps, gs)) / plen for gs, ge in gold),
                default=0.0,
            )
            prec_num += best
            prec_den += 1

        for gs, ge in gold:
            glen = ge - gs
            if glen == 0:
                continue
            best = max(
                (max(0, min(pe, ge) - max(ps, gs)) / glen for ps, pe in pred),
                default=0.0,
            )
            rec_num += best
            rec_den += 1

    P = prec_num / prec_den if prec_den else 0.0
    R = rec_num  / rec_den  if rec_den  else 0.0
    F = 2*P*R/(P+R) if (P+R) else 0.0
    return P, R, F, prec_num, prec_den, rec_num, rec_den


# ────────────────────────────────────────────────────────────────────────────
# 4. SENTENCE-LEVEL SCORING
# ────────────────────────────────────────────────────────────────────────────
def sentence_level_eval(gold_by_article, pred_by_article, texts):
    """
    A gold span is 'caught' at sentence level if ANY prediction span contains
    (or substantially overlaps) the sentence in which the gold span lives.
    A prediction span is 'useful' if it covers at least one gold sentence.

    We measure recall at 50% sentence overlap threshold (i.e. the pred span
    must cover at least 50% of the gold span's sentence to count as a hit).
    """
    THRESHOLD = 0.5   # fraction of gold sentence that pred must cover

    per_article = {}
    total_gold  = 0
    total_caught = 0
    total_pred  = 0
    total_useful = 0

    all_aids = set(gold_by_article) | set(pred_by_article)

    for aid in all_aids:
        gold = gold_by_article.get(aid, [])
        pred = pred_by_article.get(aid, [])
        text = texts.get(aid, "")
        if not text:
            continue

        sentences = split_sentences(text)

        # For each gold span: find its sentence, then check if any pred covers it
        caught_gold = []
        for gs, ge in gold:
            sent = find_sentence_for_span(gs, ge, sentences)
            if sent is None:
                caught_gold.append(False)
                continue
            ss, se = sent
            sent_len = se - ss
            covered = max(
                max(0, min(pe, se) - max(ps, ss))
                for ps, pe in pred
            ) if pred else 0
            coverage_ratio = covered / sent_len if sent_len > 0 else 0
            caught_gold.append(coverage_ratio >= THRESHOLD)

        # For each pred span: check if it covers any gold sentence
        useful_pred = []
        for ps, pe in pred:
            hit = False
            for gs, ge in gold:
                sent = find_sentence_for_span(gs, ge, sentences)
                if sent is None:
                    continue
                ss, se = sent
                sent_len = se - ss
                overlap = max(0, min(pe, se) - max(ps, ss))
                if sent_len > 0 and (overlap / sent_len) >= THRESHOLD:
                    hit = True
                    break
            useful_pred.append(hit)

        per_article[aid] = {
            "gold_n":   len(gold),
            "pred_n":   len(pred),
            "caught_n": sum(caught_gold),
            "useful_n": sum(useful_pred),
            "caught":   caught_gold,
            "useful":   useful_pred,
        }

        total_gold   += len(gold)
        total_caught += sum(caught_gold)
        total_pred   += len(pred)
        total_useful += sum(useful_pred)

    sent_recall    = total_caught / total_gold  if total_gold  else 0.0
    sent_precision = total_useful / total_pred  if total_pred  else 0.0
    sent_f1 = (2 * sent_precision * sent_recall /
               (sent_precision + sent_recall)
               if (sent_precision + sent_recall) else 0.0)

    return {
        "precision": sent_precision,
        "recall":    sent_recall,
        "f1":        sent_f1,
        "total_gold": total_gold,
        "total_caught": total_caught,
        "total_pred": total_pred,
        "total_useful": total_useful,
        "per_article": per_article,
        "threshold": THRESHOLD,
    }


# ────────────────────────────────────────────────────────────────────────────
# 5. SPAN LENGTH + DISTRIBUTION ANALYSIS
# ────────────────────────────────────────────────────────────────────────────
def span_length_stats(spans_by_article, texts):
    lengths = []
    texts_list = []
    for aid, spans in spans_by_article.items():
        text = texts.get(aid, "")
        for s, e in spans:
            l = e - s
            if l > 0:
                lengths.append(l)
                texts_list.append(text[s:e].strip())
    return lengths, texts_list


def bucket(lengths):
    bins = [(1,5),(6,10),(11,20),(21,50),(51,100),(101,300),(301,9999)]
    labels = ["1–5","6–10","11–20","21–50","51–100","101–300","300+"]
    counts = []
    for lo, hi in bins:
        counts.append(sum(1 for l in lengths if lo <= l <= hi))
    return labels, counts


# ────────────────────────────────────────────────────────────────────────────
# 6. PER-SPAN VERDICT (did LLM catch this at char level vs sentence level?)
# ────────────────────────────────────────────────────────────────────────────
def per_span_verdict(gold_by_article, pred_by_article, texts):
    """
    For every gold span classify it as:
      - exact_match   : pred span exactly equals gold span
      - char_overlap  : pred overlaps gold (partial, both ways)
      - sent_covered  : sentence is covered but not char-overlapping
      - missed        : no pred overlaps or covers the sentence
    """
    results = []
    for aid in sorted(gold_by_article):
        gold = gold_by_article.get(aid, [])
        pred = pred_by_article.get(aid, [])
        text = texts.get(aid, "")
        sentences = split_sentences(text) if text else []

        for gs, ge in gold:
            span_text = text[gs:ge].strip() if text else ""
            span_len  = ge - gs

            # check exact match
            if (gs, ge) in pred:
                verdict = "exact_match"
            else:
                # any char overlap?
                overlapping = [(ps, pe) for ps, pe in pred
                               if max(0, min(pe, ge) - max(ps, gs)) > 0]
                if overlapping:
                    verdict = "char_overlap"
                else:
                    # sentence level
                    sent = find_sentence_for_span(gs, ge, sentences)
                    if sent:
                        ss, se = sent
                        sent_cov = max(
                            (max(0, min(pe, se) - max(ps, ss)) for ps, pe in pred),
                            default=0
                        )
                        sent_len = se - ss
                        if sent_len > 0 and sent_cov / sent_len >= 0.5:
                            verdict = "sent_covered"
                        else:
                            verdict = "missed"
                    else:
                        verdict = "missed"

            results.append({
                "article_id": aid,
                "gold_start": gs,
                "gold_end": ge,
                "span_len": span_len,
                "span_text": span_text[:120],
                "verdict": verdict,
            })

    return results


# ────────────────────────────────────────────────────────────────────────────
# 7. PRINT SUMMARY
# ────────────────────────────────────────────────────────────────────────────
def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ────────────────────────────────────────────────────────────────────────────
# 8. PLOTS
# ────────────────────────────────────────────────────────────────────────────
def plot_scoring_comparison(official, sentence, out_path):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    ax.set_facecolor(PANEL)

    metrics = ["Precision", "Recall", "F1"]
    off_vals  = [official[0], official[1], official[2]]
    sent_vals = [sentence["precision"], sentence["recall"], sentence["f1"]]

    x = np.arange(3)
    w = 0.32
    b1 = ax.bar(x - w/2, off_vals,  w, color=ORANGE+"CC", label="Official partial-overlap",  zorder=3)
    b2 = ax.bar(x + w/2, sent_vals, w, color=ACCENT,      label="Sentence-level (this analysis)", zorder=3)

    for b, v in list(zip(b1, off_vals)) + list(zip(b2, sent_vals)):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=12, color=WHITE, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(metrics, color=WHITE, fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", color=LGRAY)
    ax.set_title("LLM Performance: Official vs Sentence-Level Evaluation", color=WHITE, fontsize=14)
    ax.tick_params(colors=WHITE)
    ax.grid(axis="y", color=MGRAY, alpha=0.4, zorder=0)
    for sp in ax.spines.values(): sp.set_color(MGRAY)
    ax.legend(facecolor=PANEL, edgecolor=MGRAY, labelcolor=WHITE, fontsize=11)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def plot_span_length_distribution(gold_lens, llm_lens, out_path):
    labels, gc = bucket(gold_lens)
    _,      lc = bucket(llm_lens)

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
    ax.set_facecolor(PANEL)
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, gc, w, color=GREEN,        label="Gold spans",   zorder=3)
    ax.bar(x + w/2, lc, w, color=ORANGE+"CC",  label="LLM preds",    zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels, color=WHITE, fontsize=11)
    ax.set_xlabel("Span length (characters)", color=LGRAY, fontsize=11)
    ax.set_ylabel("Count", color=LGRAY)
    ax.set_title("Span Length Distribution: Gold vs LLM", color=WHITE, fontsize=14)
    ax.tick_params(colors=WHITE)
    ax.grid(axis="y", color=MGRAY, alpha=0.4, zorder=0)
    for sp in ax.spines.values(): sp.set_color(MGRAY)
    ax.legend(facecolor=PANEL, edgecolor=MGRAY, labelcolor=WHITE, fontsize=11)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def plot_verdict_breakdown(verdicts, out_path):
    cats   = ["exact_match", "char_overlap", "sent_covered", "missed"]
    labels = ["Exact Match", "Char Overlap", "Sent Covered\n(LLM upside)", "Missed"]
    colors = [GREEN, ACCENT, PURPLE, ORANGE]

    counts = {c: sum(1 for v in verdicts if v["verdict"] == c) for c in cats}
    total  = sum(counts.values())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)

    # bar chart
    ax = axes[0]
    ax.set_facecolor(PANEL)
    ys = [counts[c] for c in cats]
    bars = ax.bar(range(4), ys, color=colors, zorder=3, edgecolor="none")
    for b, v, pct in zip(bars, ys, [v/total*100 for v in ys]):
        ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v}\n({pct:.1f}%)",
                ha="center", fontsize=10, color=WHITE, fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, color=WHITE, fontsize=10)
    ax.set_ylabel("Gold spans", color=LGRAY)
    ax.set_title("What Happened to Each Gold Span?", color=WHITE, fontsize=12)
    ax.tick_params(colors=WHITE)
    ax.grid(axis="y", color=MGRAY, alpha=0.4, zorder=0)
    for sp in ax.spines.values(): sp.set_color(MGRAY)

    # pie / donut — "found at sentence level" vs missed
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    found = counts["exact_match"] + counts["char_overlap"] + counts["sent_covered"]
    missed = counts["missed"]
    wedge_sizes = [
        counts["exact_match"],
        counts["char_overlap"],
        counts["sent_covered"],
        missed,
    ]
    wedge_colors = [GREEN, ACCENT, PURPLE, ORANGE]
    wedge_labels = [
        f"Exact ({counts['exact_match']})",
        f"Char overlap ({counts['char_overlap']})",
        f"Sentence covered ({counts['sent_covered']})",
        f"Missed ({missed})",
    ]
    wedges, texts, autotexts = ax2.pie(
        wedge_sizes, labels=wedge_labels, colors=wedge_colors,
        autopct="%1.1f%%", startangle=140,
        textprops={"color": WHITE, "fontsize": 10},
        pctdistance=0.75, labeldistance=1.12,
    )
    for at in autotexts:
        at.set_color(WHITE)
        at.set_fontweight("bold")
    ax2.set_title("Gold Span Coverage Breakdown", color=WHITE, fontsize=12)

    found_pct = found / total * 100 if total else 0
    ax2.text(0, -1.5,
             f"Total found at sentence level: {found}/{total} = {found_pct:.1f}%",
             ha="center", fontsize=11, color=GREEN, fontweight="bold",
             transform=ax2.transData)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def plot_length_vs_verdict(verdicts, out_path):
    """Show: at which span lengths does the LLM succeed/fail?"""
    by_verdict = defaultdict(list)
    for v in verdicts:
        by_verdict[v["verdict"]].append(v["span_len"])

    bins = np.array([0, 5, 10, 20, 50, 100, 300, 1000])
    labels = ["1–5","6–10","11–20","21–50","51–100","101–300","300+"]
    cats   = ["exact_match", "char_overlap", "sent_covered", "missed"]
    colors = [GREEN, ACCENT, PURPLE, ORANGE]
    cat_labels = ["Exact", "Char Overlap", "Sent Covered", "Missed"]

    # stack per bin
    bin_counts = np.zeros((len(bins)-1, len(cats)))
    for j, c in enumerate(cats):
        lens = by_verdict[c]
        for i in range(len(bins)-1):
            bin_counts[i, j] = sum(1 for l in lens if bins[i] < l <= bins[i+1])

    fig, ax = plt.subplots(figsize=(13, 5), facecolor=BG)
    ax.set_facecolor(PANEL)
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for j, (c, col, lbl) in enumerate(zip(cats, colors, cat_labels)):
        ax.bar(x, bin_counts[:, j], bottom=bottom, color=col,
               label=lbl, zorder=3, edgecolor="none")
        bottom += bin_counts[:, j]

    ax.set_xticks(x); ax.set_xticklabels(labels, color=WHITE, fontsize=11)
    ax.set_xlabel("Gold span length (characters)", color=LGRAY)
    ax.set_ylabel("Count", color=LGRAY)
    ax.set_title("LLM Coverage by Gold Span Length", color=WHITE, fontsize=14)
    ax.tick_params(colors=WHITE)
    ax.grid(axis="y", color=MGRAY, alpha=0.4, zorder=0)
    for sp in ax.spines.values(): sp.set_color(MGRAY)
    ax.legend(facecolor=PANEL, edgecolor=MGRAY, labelcolor=WHITE, fontsize=10,
              loc="upper right")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def plot_per_article_recall(sentence_result, out_path):
    pa = sentence_result["per_article"]
    aids  = sorted(pa.keys())
    off_recall = []
    sent_recall = []

    gold_by_art  = load_labels(GOLD_FILE)
    llm_by_art   = load_labels(LLM_FILE)

    for aid in aids:
        g = gold_by_art.get(aid, [])
        p = llm_by_art.get(aid, [])
        # official recall for this article
        if g:
            load_articles(ART_DIR)
            r_num = sum(
                max((max(0, min(pe, ge) - max(ps, gs)) / (ge - gs)
                     for ps, pe in p), default=0.0)
                for gs, ge in g if ge > gs
            )
            off_recall.append(r_num / len(g))
        else:
            off_recall.append(0.0)
        d = pa[aid]
        sent_recall.append(d["caught_n"] / d["gold_n"] if d["gold_n"] else 0.0)

    fig, ax = plt.subplots(figsize=(14, 5), facecolor=BG)
    ax.set_facecolor(PANEL)
    x = np.arange(len(aids))
    w = 0.38
    ax.bar(x - w/2, off_recall,  w, color=ORANGE+"CC", label="Official recall", zorder=3)
    ax.bar(x + w/2, sent_recall, w, color=ACCENT,      label="Sentence recall", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([a[-4:] for a in aids], rotation=45, color=WHITE, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Recall", color=LGRAY)
    ax.set_title("Per-Article Recall: Official vs Sentence-Level", color=WHITE, fontsize=13)
    ax.tick_params(colors=WHITE)
    ax.grid(axis="y", color=MGRAY, alpha=0.4, zorder=0)
    for sp in ax.spines.values(): sp.set_color(MGRAY)
    ax.legend(facecolor=PANEL, edgecolor=MGRAY, labelcolor=WHITE, fontsize=10)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    gold = load_labels(GOLD_FILE)
    llm  = load_labels(LLM_FILE)
    texts = load_articles(ART_DIR)

    sum(len(v) for v in gold.values())
    sum(len(v) for v in llm.values())

    # ── 1. Official partial-overlap ─────────────────────────────────────────
    P, R, F, pn, pd_, rn, rd = partial_overlap_f1(gold, llm)
    print_section("1. Official Partial-Overlap Score (character level)")
    print(f"  Precision : {pn:.3f} / {pd_} = {P:.6f}")
    print(f"  Recall    : {rn:.3f} / {rd} = {R:.6f}")
    print(f"  F1        : {F:.6f}")
    print(f"\n  (This is the score already reported: F1=0.389)")

    # ── 2. Sentence-level evaluation ───────────────────────────────────────
    print_section("2. Sentence-Level Evaluation")
    sent = sentence_level_eval(gold, llm, texts)
    print(f"  Threshold for sentence coverage : {sent['threshold']*100:.0f}%")
    print(f"  Gold spans total : {sent['total_gold']}")
    print(f"  Gold spans whose sentence was predicted (recall) : "
          f"{sent['total_caught']} / {sent['total_gold']} = {sent['recall']:.4f}")
    print(f"  LLM predictions that covered a gold sentence (precision) : "
          f"{sent['total_useful']} / {sent['total_pred']} = {sent['precision']:.4f}")
    print(f"  Sentence-level F1 : {sent['f1']:.4f}")
    print(f"\n  Interpretation: If we give the LLM full credit for identifying")
    print(f"  the sentence containing propaganda (not exact boundaries),")
    print(f"  the recall improves from {R:.3f} → {sent['recall']:.3f} "
          f"(+{sent['recall']-R:.3f}) and F1 from {F:.3f} → {sent['f1']:.3f}.")

    # ── 3. Per-span verdict ─────────────────────────────────────────────────
    print_section("3. Per-Span Verdict Breakdown")
    verdicts = per_span_verdict(gold, llm, texts)
    cat_counts = defaultdict(int)
    for v in verdicts:
        cat_counts[v["verdict"]] += 1
    print(f"  Total gold spans : {len(verdicts)}")
    for c, lbl in [("exact_match",  "Exact match      "),
                   ("char_overlap",  "Char overlap     "),
                   ("sent_covered",  "Sentence covered "),
                   ("missed",        "Missed entirely  ")]:
        n   = cat_counts[c]
        pct = n / len(verdicts) * 100 if verdicts else 0
        print(f"  {lbl}: {n:4d}  ({pct:5.1f}%)")

    found  = cat_counts["exact_match"] + cat_counts["char_overlap"] + cat_counts["sent_covered"]
    missed = cat_counts["missed"]
    print(f"\n  Found at some level  : {found} / {len(verdicts)} = "
          f"{found/len(verdicts)*100:.1f}%")
    print(f"  Completely missed    : {missed} / {len(verdicts)} = "
          f"{missed/len(verdicts)*100:.1f}%")

    # ── 4. Short-span deep dive ─────────────────────────────────────────────
    print_section("4. Short Span Deep Dive (<= 20 chars)")
    short = [v for v in verdicts if v["span_len"] <= 20]
    if short:
        sc = defaultdict(int)
        for v in short: sc[v["verdict"]] += 1
        print(f"  Short gold spans total : {len(short)}")
        for c in ["exact_match","char_overlap","sent_covered","missed"]:
            pct = sc[c] / len(short) * 100 if short else 0
            print(f"    {c:20s}: {sc[c]:3d} ({pct:.1f}%)")
        print(f"\n  Short spans caught at sentence level : "
              f"{sc['sent_covered']} ({sc['sent_covered']/len(short)*100:.1f}%)")
        print(f"  → These are 1-2 word propaganda phrases the LLM included")
        print(f"    in a larger sentence prediction — it 'saw' them but")
        print(f"    over-predicted the boundary.")

    # ── 5. Long-span deep dive ──────────────────────────────────────────────
    print_section("5. Long Span Deep Dive (>= 50 chars)")
    long_ = [v for v in verdicts if v["span_len"] >= 50]
    if long_:
        lc = defaultdict(int)
        for v in long_: lc[v["verdict"]] += 1
        print(f"  Long gold spans total : {len(long_)}")
        for c in ["exact_match","char_overlap","sent_covered","missed"]:
            pct = lc[c] / len(long_) * 100 if long_ else 0
            print(f"    {c:20s}: {lc[c]:3d} ({pct:.1f}%)")

    # ── 6. Sample sent_covered spans ────────────────────────────────────────
    print_section("6. Examples: Sentence-Covered Spans (LLM found sentence, missed exact span)")
    sent_covered_examples = [v for v in verdicts if v["verdict"] == "sent_covered"][:8]
    for ex in sent_covered_examples:
        print(f"\n  Article {ex['article_id']}  chars {ex['gold_start']}–{ex['gold_end']}"
              f"  len={ex['span_len']}")
        print(f"  Gold span: \"{ex['span_text'][:80]}\"")

    # ── 7. Sample missed spans ──────────────────────────────────────────────
    print_section("7. Examples: Completely Missed Spans (not even at sentence level)")
    missed_examples = [v for v in verdicts if v["verdict"] == "missed"][:8]
    for ex in missed_examples:
        print(f"\n  Article {ex['article_id']}  chars {ex['gold_start']}–{ex['gold_end']}"
              f"  len={ex['span_len']}")
        print(f"  Gold span: \"{ex['span_text'][:80]}\"")

    # ── 8. Plots ─────────────────────────────────────────────────────────────
    print_section("8. Generating Plots")

    plot_scoring_comparison(
        (P, R, F), sent,
        os.path.join(OUT_DIR, "plot1_scoring_comparison.png")
    )
    print("  saved plot1_scoring_comparison.png")

    gold_lens, _ = span_length_stats(gold, texts)
    llm_lens,  _ = span_length_stats(llm,  texts)
    plot_span_length_distribution(
        gold_lens, llm_lens,
        os.path.join(OUT_DIR, "plot2_span_length_distribution.png")
    )
    print("  saved plot2_span_length_distribution.png")

    plot_verdict_breakdown(
        verdicts,
        os.path.join(OUT_DIR, "plot3_verdict_breakdown.png")
    )
    print("  saved plot3_verdict_breakdown.png")

    plot_length_vs_verdict(
        verdicts,
        os.path.join(OUT_DIR, "plot4_length_vs_verdict.png")
    )
    print("  saved plot4_length_vs_verdict.png")

    plot_per_article_recall(
        sent,
        os.path.join(OUT_DIR, "plot5_per_article_recall.png")
    )
    print("  saved plot5_per_article_recall.png")

    # ── 9. Save full verdict table ───────────────────────────────────────────
    import csv
    csv_path = os.path.join(OUT_DIR, "verdict_per_span.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["article_id","gold_start","gold_end",
                                           "span_len","verdict","span_text"])
        w.writeheader()
        w.writerows(verdicts)
    print(f"  saved verdict_per_span.csv  ({len(verdicts)} rows)")

    # ── Summary ──────────────────────────────────────────────────────────────
    print_section("FINAL SUMMARY")
    print(f"""
  Official scoring (partial char overlap):
    P={P:.3f}  R={R:.3f}  F1={F:.3f}

  Sentence-level scoring (did LLM cover the right sentence?):
    P={sent['precision']:.3f}  R={sent['recall']:.3f}  F1={sent['f1']:.3f}

  Verdict breakdown over {len(verdicts)} gold spans:
    Exact match      : {cat_counts['exact_match']:4d} ({cat_counts['exact_match']/len(verdicts)*100:.1f}%)
    Char overlap     : {cat_counts['char_overlap']:4d} ({cat_counts['char_overlap']/len(verdicts)*100:.1f}%)
    Sentence covered : {cat_counts['sent_covered']:4d} ({cat_counts['sent_covered']/len(verdicts)*100:.1f}%)
    Completely missed: {cat_counts['missed']:4d} ({cat_counts['missed']/len(verdicts)*100:.1f}%)

  Key finding: The LLM actually 'saw' the propaganda sentence for
  {found}/{len(verdicts)} = {found/len(verdicts)*100:.1f}% of gold spans —
  but the standard scorer penalises it for predicting too broadly.
  The {cat_counts['sent_covered']} 'sentence covered' spans are cases where
  the LLM's instinct was correct but its boundary was wrong.
    """)


if __name__ == "__main__":
    main()
