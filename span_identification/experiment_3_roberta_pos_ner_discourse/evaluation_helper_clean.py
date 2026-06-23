# Run a trained span model on the dev articles and score it with the official scorer.
import os
import torch
import pandas as pd
import subprocess
import tempfile
import spacy
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from model import (
    TransformerSpanTagger,
    PTCSpanDataset,
    collate_fn,
    decode_bioes_token_spans,
    merge_overlapping_spans,
    CFG,
    NUM_LABELS
)

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt["cfg"], dict):
        cfg_dict = ckpt["cfg"]
        valid_keys = CFG.__dataclass_fields__.keys()
        filtered_cfg = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        cfg = CFG(**filtered_cfg)
    else:
        cfg = ckpt["cfg"]

    pos_vocab = ckpt["pos_vocab"]
    ner_vocab = ckpt["ner_vocab"]

    model = TransformerSpanTagger(
        model_name=cfg.model_name,
        num_labels=NUM_LABELS,
        num_pos_tags=len(pos_vocab),
        num_ner_tags=len(ner_vocab),
        pos_dim=cfg.pos_dim,
        ner_dim=cfg.ner_dim,
        discourse_input_dim=11,
        discourse_dim=cfg.discourse_dim,
        lstm_hidden=cfg.lstm_hidden,
        dropout=cfg.dropout
    )

    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    return model, tokenizer, cfg, pos_vocab, ner_vocab


def load_spacy():
    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer", "textcat"])
    if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def build_df_from_articles(folder):
    rows = []

    for file in os.listdir(folder):
        if file.endswith(".txt"):
            aid = file.replace("article", "").replace(".txt", "")
            text = open(os.path.join(folder, file), encoding="utf-8").read()

            rows.append({
                "article_id": str(aid),
                "article_text": text
            })

    return pd.DataFrame(rows)


def build_unlabeled_records(df):
    return [
        {
            "article_id": str(row["article_id"]),
            "article_text": row["article_text"],
            "gold_spans": []
        }
        for _, row in df.iterrows()
    ]


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
        discourse_feats = batch["discourse_feats"].to(device)

        preds = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            crf_mask=crf_mask,
            pos_ids=pos_ids,
            ner_ids=ner_ids,
            discourse_feats=discourse_feats,
            labels=None
        )

        for i in range(len(preds)):
            aid = batch["article_id"][i]
            offsets = batch["offset_mapping"][i]

            spans = decode_bioes_token_spans(preds[i], offsets)
            pred_spans_by_article_raw[aid].extend(spans)

    return {
        aid: merge_overlapping_spans(spans)
        for aid, spans in pred_spans_by_article_raw.items()
    }


def save_predictions(pred_spans, output_file):
    with open(output_file, "w") as f:
        for aid in sorted(pred_spans.keys()):
            for s, e in pred_spans[aid]:
                if s < e:
                    f.write(f"{aid}\t{s}\t{e}\n")


def run_official_scorer(script, pred_file, gold_file):
    result = subprocess.run(
        ["python", script, "-s", pred_file, "-r", gold_file],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print("ERROR:", result.stderr)


def predict_and_score_with_official_scorer(
    checkpoint_path,
    articles_folder,
    gold_labels_file,
    scorer_script_path
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer, cfg, pos_vocab, ner_vocab = load_model(
        checkpoint_path, device
    )

    nlp = load_spacy()

    df = build_df_from_articles(articles_folder)
    records = build_unlabeled_records(df)

    ds = PTCSpanDataset(
        records,
        tokenizer,
        nlp,
        pos_vocab,
        ner_vocab,
        max_length=cfg.max_length,
        stride=cfg.stride,
        is_train=False
    )

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    pred_spans = predict_spans(model, loader, device)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".labels")
    pred_file = tmp.name
    tmp.close()

    save_predictions(pred_spans, pred_file)

    run_official_scorer(
        scorer_script_path,
        pred_file,
        gold_labels_file
    )

    return pred_spans