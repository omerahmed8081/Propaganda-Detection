# evaluation_helper_clean.py

import os
import subprocess
import tempfile
import torch
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer

# 🔥 import from your notebook file
from model import (
    DebertaSpanTagger,
    PTCSpanDataset,
    collate_fn,
    decode_bioes_token_spans,
    merge_overlapping_spans,
    token_level_f1,
    exact_span_f1,
    build_article_records,
    build_gold_spans_by_article,
    LABEL2ID,
    NUM_LABELS,
    CFG
)


# =========================================================
# LOAD MODEL
# =========================================================
def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    # 🔥 FIX HERE
    if isinstance(ckpt["cfg"], dict):
        cfg = CFG(**ckpt["cfg"])
    else:
        cfg = ckpt["cfg"]

    pos_vocab = ckpt.get("pos_vocab", {"<PAD>": 0})
    ner_vocab = ckpt.get("ner_vocab", {"<PAD>": 0})

    model = DebertaSpanTagger(
        model_name=cfg.model_name,
        num_labels=NUM_LABELS,
        num_pos_tags=len(pos_vocab),
        num_ner_tags=len(ner_vocab),
        pos_dim=cfg.pos_dim,
        ner_dim=cfg.ner_dim,
        discourse_dim=0,
        lstm_hidden=cfg.lstm_hidden,
        dropout=cfg.dropout
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    return model, tokenizer, cfg, pos_vocab, ner_vocab


# =========================================================
# PREDICT SPANS
# =========================================================
@torch.no_grad()
def predict_spans(model, loader, device):
    model.eval()

    pred_spans_by_article_raw = defaultdict(list)

    for batch in tqdm(loader, desc="Predicting"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        crf_mask = batch["crf_mask"].to(device)
        pos_ids = batch["pos_ids"].to(device)
        ner_ids = batch["ner_ids"].to(device)

        preds = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            crf_mask=crf_mask,
            pos_ids=pos_ids,
            ner_ids=ner_ids,
            labels=None
        )

        for i in range(len(preds)):
            article_id = batch["article_id"][i]
            offsets = batch["offset_mapping"][i]

            spans = decode_bioes_token_spans(preds[i], offsets)
            pred_spans_by_article_raw[article_id].extend(spans)

    # merge overlapping spans
    pred_spans_by_article = {
        aid: merge_overlapping_spans(spans)
        for aid, spans in pred_spans_by_article_raw.items()
    }

    return pred_spans_by_article


# =========================================================
# DATA PREP FROM FOLDER
# =========================================================
def build_df_from_articles(articles_folder):
    rows = []

    for file in os.listdir(articles_folder):
        if file.endswith(".txt"):
            aid = file.replace("article", "").replace(".txt", "")
            text = open(os.path.join(articles_folder, file), encoding="utf-8").read()

            rows.append({
                "article_id": str(aid),
                "article_text": text
            })

    return pd.DataFrame(rows)


# =========================================================
# SAVE PREDICTIONS
# =========================================================
def save_predictions(pred_spans, output_file):
    with open(output_file, "w") as f:
        for aid in sorted(pred_spans.keys()):
            for s, e in pred_spans[aid]:
                if s < e:
                    f.write(f"{aid}\t{s}\t{e}\n")


# =========================================================
# RUN OFFICIAL SCORER
# =========================================================
def run_official_scorer(scorer_script, pred_file, gold_file):
    result = subprocess.run(
        ["python", scorer_script, "-s", pred_file, "-r", gold_file],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print("ERROR:", result.stderr)

    return result


# =========================================================
# MAIN FUNCTION (WHAT YOU WANT)
# =========================================================
def predict_and_score_with_official_scorer(
    checkpoint_path,
    articles_folder,
    gold_labels_file,
    scorer_script_path
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model, tokenizer, cfg, pos_vocab, ner_vocab = load_model(
        checkpoint_path, device
    )

    # build dataset
    df = build_df_from_articles(articles_folder)
    records = [
    {
        "article_id": str(row["article_id"]),
        "article_text": row["article_text"],
        "gold_spans": []   # 🔥 IMPORTANT: no labels
    }
    for _, row in df.iterrows()
]

    ds = PTCSpanDataset(
        records,
        tokenizer,
        nlp=None,
        pos_vocab=pos_vocab,
        ner_vocab=ner_vocab,
        max_length=cfg.max_length,
        stride=cfg.stride,
        is_train=False
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # predict
    pred_spans = predict_spans(model, loader, device)

    # save temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".labels")
    pred_file = tmp.name
    tmp.close()

    save_predictions(pred_spans, pred_file)

    # run scorer
    run_official_scorer(
        scorer_script_path,
        pred_file,
        gold_labels_file
    )

    return pred_spans