"""
Extract span representations from the Experiment-2 TC model and plot
a UMAP scatter coloured by propaganda technique.

Run from Experiment_2_discourse_model/:
    python umap_by_technique.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from umap import UMAP
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from model import CFG, TechniqueDataset, parse_labels

# ── Paths ────────────────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
CKPT      = os.path.join(HERE, "best_technique_model.pt")
DATA_DIR  = "/home/omer_ahmed/Experiments/Technique_classification/processed_span_data"
OUT_DIR   = os.path.join(HERE, "umap_results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colour palette (14 techniques) ───────────────────────────────────────────
TECHNIQUE_COLORS = {
    "Loaded_Language":                      "#00B4D8",
    "Name_Calling,Labeling":               "#FF7043",
    "Doubt":                               "#26A69A",
    "Repetition":                          "#AB47BC",
    "Exaggeration,Minimisation":           "#FFA726",
    "Flag-Waving":                         "#EF5350",
    "Appeal_to_fear-prejudice":            "#66BB6A",
    "Causal_Oversimplification":           "#EC407A",
    "Whataboutism,Straw_Men,Red_Herring":  "#29B6F6",
    "Appeal_to_Authority":                 "#D4E157",
    "Bandwagon,Reductio_ad_hitlerum":      "#FF7043",
    "Slogans":                             "#8D6E63",
    "Black-and-White_Fallacy":             "#78909C",
    "Thought-terminating_Cliches":         "#F06292",
}

# Give each technique a unique colour (override duplicates above properly)
PALETTE = [
    "#00B4D8", "#FF7043", "#26A69A", "#AB47BC", "#FFA726",
    "#EF5350", "#66BB6A", "#EC407A", "#29B6F6", "#D4E157",
    "#BCAAA4", "#8D6E63", "#78909C", "#F06292",
]


# ── Model with representation hook ───────────────────────────────────────────
class TechniqueClassifierRep(nn.Module):
    """Same as TechniqueClassifier but forward() returns the rep vector."""

    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        hidden_size     = self.encoder.config.hidden_size
        self.dropout    = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size * 3, num_labels)

    def forward(self, input_ids, attention_mask, span_start, span_end):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden  = outputs.last_hidden_state
        cls     = hidden[:, 0]
        B       = hidden.size(0)

        span_means, span_maxs = [], []
        for i in range(B):
            s, e = span_start[i].item(), span_end[i].item()
            if e <= s:
                e = s + 1
            toks = hidden[i, s:e]
            span_means.append(toks.mean(dim=0))
            span_maxs.append(toks.max(dim=0).values)

        span_means = torch.stack(span_means)
        span_maxs  = torch.stack(span_maxs)
        rep        = torch.cat([cls, span_means, span_maxs], dim=1)
        return rep   # (B, hidden*3)  — no classifier applied


# ── Extract representations ───────────────────────────────────────────────────
@torch.no_grad()
def extract_reps(model, dataset, batch_size=16):
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    all_reps = []

    for batch in tqdm(loader, desc="Extracting"):
        rep = model(
            batch["input_ids"].to(CFG.device),
            batch["attention_mask"].to(CFG.device),
            batch["span_start"].to(CFG.device),
            batch["span_end"].to(CFG.device),
        )
        all_reps.append(rep.cpu().float().numpy())

    return np.vstack(all_reps)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # 1. Load val data with gold techniques
    val_df = pd.read_parquet(os.path.join(DATA_DIR, "val.parquet"))
    val_df["techniques"] = val_df["techniques"].apply(parse_labels)

    # 2. Load checkpoint
    ckpt     = torch.load(CKPT, map_location=CFG.device)
    label2id = ckpt["label2id"]
    ckpt["id2label"]
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    # 3. Build dataset (reuses Experiment-2's TechniqueDataset)
    dataset = TechniqueDataset(val_df, tokenizer, label2id)

    # Collect per-span primary technique (99.9% of spans have exactly 1)
    techniques_per_span = []
    for _, row in val_df.iterrows():
        techs = row["techniques"]
        techniques_per_span.append(techs[0] if techs else "Unknown")

    # 4. Build model and load weights
    model = TechniqueClassifierRep(CFG.model_name, len(label2id)).to(CFG.device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    # 5. Extract 3072-dim representations
    print(f"\nExtracting representations for {len(dataset)} spans...")
    reps = extract_reps(model, dataset)
    print(f"Representation matrix: {reps.shape}")

    # 6. UMAP → 2D
    print("\nRunning UMAP...")
    reducer = UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                   metric="cosine", random_state=42)
    emb = reducer.fit_transform(reps)
    print("UMAP done.")

    # 7. Assign colours
    all_techs = sorted(set(techniques_per_span))
    tech2color = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(all_techs)}
    [tech2color[t] for t in techniques_per_span]

    # 8. Save embeddings + labels for reuse
    df_out = pd.DataFrame({
        "umap_x":    emb[:, 0],
        "umap_y":    emb[:, 1],
        "technique": techniques_per_span,
        "span_text": [s["text"][:80] for s in dataset.samples],
    })
    df_out.to_csv(os.path.join(OUT_DIR, "umap_technique_embeddings.csv"), index=False)

    # 9. Plot ─────────────────────────────────────────────────────────────────
    BG, PANEL = "#0D1B2A", "#112233"

    fig, ax = plt.subplots(figsize=(14, 9), facecolor=BG)
    ax.set_facecolor(BG)

    # Draw each technique as its own scatter layer (so legend is clean)
    for tech in all_techs:
        mask = np.array(techniques_per_span) == tech
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c=tech2color[tech],
            s=18, alpha=0.75, linewidths=0,
            label=tech,
            rasterized=True,
        )

    # Legend
    legend = ax.legend(
        loc="upper left",
        fontsize=8.5,
        framealpha=0.85,
        facecolor=PANEL,
        edgecolor="#546E7A",
        labelcolor="#B0BEC5",
        markerscale=1.8,
        title="Propaganda Technique",
        title_fontsize=9,
    )
    legend.get_title().set_color("#FFFFFF")

    ax.set_title(
        "UMAP of Span Representations — Coloured by Propaganda Technique\n"
        "(Experiment 2 · RoBERTa-large · CLS + span_mean + span_max · val set)",
        fontsize=13, color="#FFFFFF", pad=12,
    )
    ax.set_xlabel("UMAP dim 1", color="#B0BEC5", fontsize=10)
    ax.set_ylabel("UMAP dim 2", color="#B0BEC5", fontsize=10)
    ax.tick_params(colors="#546E7A")
    for spine in ax.spines.values():
        spine.set_color("#546E7A")

    # Technique count annotation
    count_str = "  ".join(
        f"{t.split(',')[0]}: {(np.array(techniques_per_span)==t).sum()}"
        for t in all_techs
    )
    fig.text(0.5, 0.01, count_str, ha="center", fontsize=7.5,
             color="#546E7A", wrap=True)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out_path = os.path.join(OUT_DIR, "umap_by_technique.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\n✅ Saved: {out_path}")

    # 10. Second plot: small multiples — one panel per technique ──────────────
    n_techs = len(all_techs)
    ncols   = 4
    nrows   = (n_techs + ncols - 1) // ncols

    fig2, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.2),
                              facecolor=BG)
    axes_flat = axes.flatten()

    for i, tech in enumerate(all_techs):
        ax2 = axes_flat[i]
        ax2.set_facecolor(BG)
        mask = np.array(techniques_per_span) == tech

        # All other spans in dim grey
        ax2.scatter(emb[~mask, 0], emb[~mask, 1],
                    c="#1E2D3D", s=6, alpha=0.4, linewidths=0, rasterized=True)
        # Highlighted technique
        ax2.scatter(emb[mask, 0], emb[mask, 1],
                    c=tech2color[tech], s=18, alpha=0.9,
                    linewidths=0, rasterized=True)

        short = tech.split(",")[0]
        ax2.set_title(f"{short}\n(n={mask.sum()})",
                      fontsize=8.5, color="#FFFFFF", pad=4)
        ax2.set_xticks([]); ax2.set_yticks([])
        for spine in ax2.spines.values():
            spine.set_color("#112233")

    # Hide empty panels
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig2.suptitle(
        "UMAP by Technique — Small Multiples  (Experiment 2 · val set)",
        fontsize=13, color="#FFFFFF", y=1.01,
    )
    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, "umap_by_technique_multiples.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig2)
    print(f"✅ Saved: {out2}")


if __name__ == "__main__":
    main()
