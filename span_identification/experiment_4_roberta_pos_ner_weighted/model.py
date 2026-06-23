# Span model with POS/NER features and a class-weighted loss.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import config

import pandas as pd
import torch
import numpy as np
import random
from dataclasses import dataclass
from collections import defaultdict
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torchcrf import CRF
import spacy
from tqdm.auto import tqdm
import logging
@dataclass
class CFG:
    model_name: str = "roberta-large"
    max_length: int = 512
    stride: int = 384
    batch_size: int = 2
    lr: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 3
    warmup_ratio: float = 0.1
    num_workers: int = 2
    seed: int = 42
    lstm_hidden: int = 512
    pos_dim: int = 32
    ner_dim: int = 32
    dropout: float = 0.2
    save_path: str = "best_span_roberta_model.pt"

    class_weights: tuple = (1.0, 3.0, 3.0, 3.0, 3.0)
    ce_loss_weight: float = 0.5
    crf_loss_weight: float = 1.0


LABEL2ID = {
    "O": 0,
    "B-PROP": 1,
    "I-PROP": 2,
    "E-PROP": 3,
    "S-PROP": 4,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_article_records(df: pd.DataFrame):
    records = []

    grouped = df.groupby("article_id", as_index=False)
    for _, g in grouped:
        article_id = str(g["article_id"].iloc[0])

        article_text = g["article_text"].iloc[0]
        article_text = "" if pd.isna(article_text) else str(article_text)

        spans = (
            g[["span_start", "span_end"]]
            .drop_duplicates()
            .sort_values(["span_start", "span_end"])
            .values.tolist()
        )

        clean_spans = []
        for s, e in spans:
            if pd.notna(s) and pd.notna(e):
                s, e = int(s), int(e)
                s = max(0, s)
                e = min(len(article_text), e)
                if s < e:
                    clean_spans.append((s, e))

        records.append({
            "article_id": article_id,
            "article_text": article_text,
            "gold_spans": clean_spans
        })

    return records
def build_pos_ner_vocab(article_records, nlp):
    """
    Build vocabularies from training articles only.
    0 is reserved for PAD/UNK.
    """
    pos_set = set()
    ner_set = set()

    for rec in tqdm(article_records, desc="Building POS/NER vocab"):
        doc = nlp(rec["article_text"])
        for tok in doc:
            pos_set.add(tok.pos_)
        for ent in doc.ents:
            ner_set.add(ent.label_)

    pos_vocab = {"<PAD>": 0}
    ner_vocab = {"<PAD>": 0}

    for i, tag in enumerate(sorted(pos_set), start=1):
        pos_vocab[tag] = i

    for i, tag in enumerate(sorted(ner_set), start=1):
        ner_vocab[tag] = i

    return pos_vocab, ner_vocab


def build_char_feature_maps(text, nlp, pos_vocab, ner_vocab):
    """
    Returns two arrays of length len(text):
      pos_char_ids[char_idx]
      ner_char_ids[char_idx]
    """
    pos_char_ids = np.zeros(len(text), dtype=np.int64)
    ner_char_ids = np.zeros(len(text), dtype=np.int64)

    doc = nlp(text)

    for tok in doc:
        pos_id = pos_vocab.get(tok.pos_, 0)
        start, end = tok.idx, tok.idx + len(tok.text)
        pos_char_ids[start:end] = pos_id

    for ent in doc.ents:
        ner_id = ner_vocab.get(ent.label_, 0)
        start, end = ent.start_char, ent.end_char
        ner_char_ids[start:end] = ner_id

    return pos_char_ids, ner_char_ids


def build_gold_char_mask(text_len, spans):
    mask = np.zeros(text_len, dtype=np.int64)
    for s, e in spans:
        s = max(0, s)
        e = min(text_len, e)
        if s < e:
            mask[s:e] = 1
    return mask


def assign_bioes_from_offsets(offsets, gold_char_mask):
    """
    offsets: list of (start, end) for tokens in one window
    gold_char_mask: array over full article chars

    Returns:
      labels: list[int] length = len(offsets)
      crf_mask: list[int] where 1 means real token, 0 means special/pad
    """
    token_is_prop = []
    crf_mask = []

    for start, end in offsets:
        if end <= start:
            token_is_prop.append(0)
            crf_mask.append(0)
        else:
            overlap = gold_char_mask[start:end].max() > 0
            token_is_prop.append(1 if overlap else 0)
            crf_mask.append(1)

    labels = [LABEL2ID["O"]] * len(offsets)

    i = 0
    n = len(offsets)
    while i < n:
        if crf_mask[i] == 0 or token_is_prop[i] == 0:
            i += 1
            continue

        j = i
        while j + 1 < n and crf_mask[j + 1] == 1 and token_is_prop[j + 1] == 1:
            j += 1

        if i == j:
            labels[i] = LABEL2ID["S-PROP"]
        else:
            labels[i] = LABEL2ID["B-PROP"]
            for k in range(i + 1, j):
                labels[k] = LABEL2ID["I-PROP"]
            labels[j] = LABEL2ID["E-PROP"]

        i = j + 1
    if len(crf_mask) > 0:
        crf_mask[0] = 1
    return labels, crf_mask


# Dataset: sliding-window tokenisation and BIOES labels
class PTCSpanDataset(Dataset):
    def __init__(self, article_records, tokenizer, nlp, pos_vocab, ner_vocab,
                 max_length=512, stride=384, is_train=True):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.is_train = is_train

        for rec in tqdm(article_records, desc="Preparing dataset"):
            article_id = rec["article_id"]
            text = rec["article_text"]
            gold_spans = rec["gold_spans"]

            gold_char_mask = build_gold_char_mask(len(text), gold_spans)
            pos_char_ids, ner_char_ids = build_char_feature_maps(text, nlp, pos_vocab, ner_vocab)

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                return_overflowing_tokens=True,
                truncation=True,
                max_length=max_length,
                stride=max_length - stride,
                padding="max_length"
            )

            for i in range(len(enc["input_ids"])):
                input_ids = enc["input_ids"][i]
                attention_mask = enc["attention_mask"][i]
                offsets = enc["offset_mapping"][i]

                labels, crf_mask = assign_bioes_from_offsets(offsets, gold_char_mask)

                pos_ids = []
                ner_ids = []

                for start, end in offsets:
                    if end <= start:
                        pos_ids.append(0)
                        ner_ids.append(0)
                    else:
                        pos_ids.append(int(pos_char_ids[start]))
                        ner_ids.append(int(ner_char_ids[start]))

                self.examples.append({
                    "article_id": article_id,
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "crf_mask": torch.tensor(crf_mask, dtype=torch.bool),
                    "pos_ids": torch.tensor(pos_ids, dtype=torch.long),
                    "ner_ids": torch.tensor(ner_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "offset_mapping": offsets,
                    "text_len": len(text),
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "article_id": ex["article_id"],
            "input_ids": ex["input_ids"],
            "attention_mask": ex["attention_mask"],
            "crf_mask": ex["crf_mask"],
            "pos_ids": ex["pos_ids"],
            "ner_ids": ex["ner_ids"],
            "labels": ex["labels"],
            "offset_mapping": ex["offset_mapping"],
            "text_len": ex["text_len"],
        }


def collate_fn(batch):
    return {
        "article_id": [x["article_id"] for x in batch],
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "crf_mask": torch.stack([x["crf_mask"] for x in batch]),
        "pos_ids": torch.stack([x["pos_ids"] for x in batch]),
        "ner_ids": torch.stack([x["ner_ids"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
        "offset_mapping": [x["offset_mapping"] for x in batch],
        "text_len": [x["text_len"] for x in batch],
    }


# Model
class DebertaSpanTagger(nn.Module):
    def __init__(
        self,
        model_name="roberta-large",
        num_labels=5,
        num_pos_tags=50,
        num_ner_tags=30,
        pos_dim=32,
        ner_dim=32,
        discourse_dim=0,
        lstm_hidden=512,
        dropout=0.1,
        class_weights=None,
        ce_loss_weight=0.5,
        crf_loss_weight=1.0
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.pos_embedding = nn.Embedding(num_pos_tags, pos_dim)
        self.ner_embedding = nn.Embedding(num_ner_tags, ner_dim)

        self.discourse_dim = discourse_dim
        fusion_input_dim = hidden_size + pos_dim + ner_dim + discourse_dim
        self.fusion = nn.Linear(fusion_input_dim, hidden_size)

        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

        nn.init.xavier_uniform_(self.fusion.weight)
        nn.init.xavier_uniform_(self.classifier.weight)

        self.ce_loss_weight = ce_loss_weight
        self.crf_loss_weight = crf_loss_weight

        if class_weights is None:
            class_weights = [1.0] * num_labels

        self.register_buffer(
            "class_weights_tensor",
            torch.tensor(class_weights, dtype=torch.float)
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        crf_mask,
        pos_ids,
        ner_ids,
        discourse_feats=None,
        labels=None
    ):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = enc.last_hidden_state

        if not torch.isfinite(token_embeddings).all():
            raise ValueError("Non-finite values detected in encoder output")

        pos_emb = self.pos_embedding(pos_ids)
        ner_emb = self.ner_embedding(ner_ids)

        if not torch.isfinite(pos_emb).all():
            raise ValueError("Non-finite values detected in POS embeddings")
        if not torch.isfinite(ner_emb).all():
            raise ValueError("Non-finite values detected in NER embeddings")

        feats = [token_embeddings, pos_emb, ner_emb]
        if discourse_feats is not None:
            if not torch.isfinite(discourse_feats).all():
                raise ValueError("Non-finite values detected in discourse features")
            feats.append(discourse_feats)

        fused = torch.cat(feats, dim=-1)

        if not torch.isfinite(fused).all():
            raise ValueError("Non-finite values detected after concatenation")

        fused = self.fusion(fused)

        if not torch.isfinite(fused).all():
            raise ValueError("Non-finite values detected after fusion layer")

        lstm_out, _ = self.bilstm(fused)

        if not torch.isfinite(lstm_out).all():
            raise ValueError("Non-finite values detected after BiLSTM")

        lstm_out = self.dropout(lstm_out)
        emissions = self.classifier(lstm_out)

        if not torch.isfinite(emissions).all():
            raise ValueError("Non-finite values detected in emissions")

        if labels is not None:
            crf_loss = -self.crf(emissions, labels, mask=crf_mask, reduction="token_mean")

            if not torch.isfinite(crf_loss):
                raise ValueError("Non-finite CRF loss detected")

            valid_mask = crf_mask.view(-1)
            flat_emissions = emissions.view(-1, emissions.size(-1))[valid_mask]
            flat_labels = labels.view(-1)[valid_mask]

            ce_loss_fct = nn.CrossEntropyLoss(weight=self.class_weights_tensor)
            ce_loss = ce_loss_fct(flat_emissions, flat_labels)

            if not torch.isfinite(ce_loss):
                raise ValueError("Non-finite CE loss detected")

            total_loss = self.crf_loss_weight * crf_loss + self.ce_loss_weight * ce_loss

            if not torch.isfinite(total_loss):
                raise ValueError("Non-finite total loss detected")

            return total_loss
        else:
            preds = self.crf.decode(emissions, mask=crf_mask)
            return preds

def token_level_f1(gold_list, pred_list):
    """
    Micro-F1 over non-padding real tokens.
    Binary evaluation:
      propaganda tags = {B,I,E,S}
      non-propaganda = O
    """
    tp = fp = fn = 0

    for gold_seq, pred_seq in zip(gold_list, pred_list):
        for g, p in zip(gold_seq, pred_seq):
            g_bin = 1 if g != LABEL2ID["O"] else 0
            p_bin = 1 if p != LABEL2ID["O"] else 0

            if g_bin == 1 and p_bin == 1:
                tp += 1
            elif g_bin == 0 and p_bin == 1:
                fp += 1
            elif g_bin == 1 and p_bin == 0:
                fn += 1

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def spans_from_char_mask(mask):
    spans = []
    in_span = False
    start = 0
    for i, val in enumerate(mask):
        if val == 1 and not in_span:
            start = i
            in_span = True
        elif val == 0 and in_span:
            spans.append((start, i))
            in_span = False
    if in_span:
        spans.append((start, len(mask)))
    return spans


def exact_span_f1(gold_spans_by_article, pred_spans_by_article):
    tp = fp = fn = 0

    all_articles = set(gold_spans_by_article.keys()) | set(pred_spans_by_article.keys())

    for aid in all_articles:
        gold = set(gold_spans_by_article.get(aid, []))
        pred = set(pred_spans_by_article.get(aid, []))

        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1

def decode_bioes_token_spans(tag_seq, offsets):
    """
    Convert one predicted BIOES tag sequence into character spans.

    Parameters
    ----------
    tag_seq : list[int]
        Predicted label ids for one window.
    offsets : list[tuple[int, int]]
        Offset mapping for one window.

    Returns
    -------
    spans : list[tuple[int, int]]
        Character spans decoded conservatively from valid BIOES structure.
    """
    spans = []
    i = 0
    n = len(tag_seq)

    while i < n:
        tag = tag_seq[i]

        if offsets[i][1] <= offsets[i][0]:
            i += 1
            continue

        if tag == LABEL2ID["O"]:
            i += 1
            continue

        if tag == LABEL2ID["S-PROP"]:
            start, end = offsets[i]
            spans.append((start, end))
            i += 1
            continue

        if tag == LABEL2ID["B-PROP"]:
            start = offsets[i][0]
            j = i + 1

            if j < n and offsets[j][1] > offsets[j][0] and tag_seq[j] == LABEL2ID["E-PROP"]:
                end = offsets[j][1]
                spans.append((start, end))
                i = j + 1
                continue

            while j < n:
                if offsets[j][1] <= offsets[j][0]:
                    break

                if tag_seq[j] == LABEL2ID["I-PROP"]:
                    j += 1
                    continue

                if tag_seq[j] == LABEL2ID["E-PROP"]:
                    end = offsets[j][1]
                    spans.append((start, end))
                    i = j + 1
                    break

                break
            else:
                pass

            if i < n and tag_seq[i] == LABEL2ID["B-PROP"]:
                i += 1
            continue

        if tag in (LABEL2ID["I-PROP"], LABEL2ID["E-PROP"]):
            i += 1
            continue

        i += 1

    return spans
def merge_overlapping_spans(spans):
    """
    Merge overlapping or touching character spans.
    """
    if not spans:
        return []

    spans = sorted(spans, key=lambda x: (x[0], x[1]))
    merged = [spans[0]]

    for s, e in spans[1:]:
        last_s, last_e = merged[-1]

        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))

    return merged
@torch.no_grad()
def predict_spans(model, loader, device):
    model.eval()

    pred_spans_by_article_raw = defaultdict(list)
    all_gold_token = []
    all_pred_token = []

    for batch in tqdm(loader, desc="Predicting"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        crf_mask = batch["crf_mask"].to(device)
        pos_ids = batch["pos_ids"].to(device)
        ner_ids = batch["ner_ids"].to(device)
        labels = batch["labels"].cpu().numpy()

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
            gold_seq = labels[i].tolist()
            pred_seq = preds[i]

            real_gold = []
            real_pred = []
            for g, p, (s, e) in zip(gold_seq, pred_seq, offsets):
                if e > s:
                    real_gold.append(g)
                    real_pred.append(p)

            all_gold_token.append(real_gold)
            all_pred_token.append(real_pred)

            window_spans = decode_bioes_token_spans(pred_seq, offsets)
            pred_spans_by_article_raw[article_id].extend(window_spans)

    pred_spans_by_article = {
        aid: merge_overlapping_spans(spans)
        for aid, spans in pred_spans_by_article_raw.items()
    }

    token_p, token_r, token_f1 = token_level_f1(all_gold_token, all_pred_token)
    return pred_spans_by_article, (token_p, token_r, token_f1)

def build_gold_spans_by_article(article_records):
    d = {}
    for rec in article_records:
        d[rec["article_id"]] = rec["gold_spans"]
    return d


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(tqdm(loader, desc="Training")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        crf_mask = batch["crf_mask"].to(device)
        pos_ids = batch["pos_ids"].to(device)
        ner_ids = batch["ner_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        if not torch.isfinite(input_ids.float()).all():
            raise ValueError(f"Non-finite input_ids at step {step}")
        if not torch.isfinite(attention_mask.float()).all():
            raise ValueError(f"Non-finite attention_mask at step {step}")
        if not torch.isfinite(pos_ids.float()).all():
            raise ValueError(f"Non-finite pos_ids at step {step}")
        if not torch.isfinite(ner_ids.float()).all():
            raise ValueError(f"Non-finite ner_ids at step {step}")

        loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            crf_mask=crf_mask,
            pos_ids=pos_ids,
            ner_ids=ner_ids,
            labels=labels
        )

        if not torch.isfinite(loss):
            print(f"Non-finite loss at step {step}")
            print("article_ids:", batch["article_id"])
            raise ValueError("NaN/Inf loss detected")

        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise ValueError(f"Non-finite gradient in {name} at step {step}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))

def evaluate(model, loader, article_records, device):
    gold_spans_by_article = build_gold_spans_by_article(article_records)
    pred_spans_by_article, (tok_p, tok_r, tok_f1) = predict_spans(model, loader, device)
    span_p, span_r, span_f1 = exact_span_f1(gold_spans_by_article, pred_spans_by_article)

    metrics = {
        "token_precision": tok_p,
        "token_recall": tok_r,
        "token_f1": tok_f1,
        "span_precision": span_p,
        "span_recall": span_r,
        "span_f1": span_f1,
    }
    return metrics, pred_spans_by_article


# Training
def run_training(train_data, val_data, cfg: CFG):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_articles = build_article_records(train_data)
    val_articles = build_article_records(val_data)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer", "textcat"])

    pos_vocab, ner_vocab = build_pos_ner_vocab(train_articles, nlp)

    train_ds = PTCSpanDataset(
        train_articles, tokenizer, nlp, pos_vocab, ner_vocab,
        max_length=cfg.max_length, stride=cfg.stride, is_train=True
    )
    val_ds = PTCSpanDataset(
        val_articles, tokenizer, nlp, pos_vocab, ner_vocab,
        max_length=cfg.max_length, stride=cfg.stride, is_train=False
    )


    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_fn
    )


    model = DebertaSpanTagger(
        model_name=cfg.model_name,
        num_labels=NUM_LABELS,
        num_pos_tags=len(pos_vocab),
        num_ner_tags=len(ner_vocab),
        pos_dim=cfg.pos_dim,
        ner_dim=cfg.ner_dim,
        discourse_dim=0,
        lstm_hidden=cfg.lstm_hidden,
        dropout=cfg.dropout,
        class_weights=cfg.class_weights,
        ce_loss_weight=cfg.ce_loss_weight,
        crf_loss_weight=cfg.crf_loss_weight
    ).to(device)

    model.float()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = int(cfg.warmup_ratio * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )


    best_val_f1 = -1
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)

        val_metrics, _ = evaluate(model, val_loader, val_articles, device)

        print(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_token_f1={val_metrics['token_f1']:.4f} | "
            f"val_span_f1={val_metrics['span_f1']:.4f}"
        )

        if val_metrics["span_f1"] > best_val_f1:
            best_val_f1 = val_metrics["span_f1"]
            best_state = {
                "model_state_dict": model.state_dict(),
                "pos_vocab": pos_vocab,
                "ner_vocab": ner_vocab,
                "cfg": cfg.__dict__,
                "best_val_span_f1": best_val_f1,
            }
            torch.save(best_state, cfg.save_path)
            print(f"Saved best model to {cfg.save_path}")

    return model, tokenizer, pos_vocab, ner_vocab, train_ds, val_ds

def main():
    logging.basicConfig(level=logging.INFO)
    print("🚀 Starting Training Pipeline...")

    train_data = pd.read_parquet(config.SPAN_TRAIN_PARQUET)
    val_data = pd.read_parquet(config.SPAN_VAL_PARQUET)
    train_data.drop(columns=["span_text"], inplace=True)
    val_data.drop(columns=["span_text"], inplace=True)

    cfg = CFG(
        model_name="roberta-large",
        max_length=512,
        stride=384,
        batch_size=2,
        lr=1e-5,
        epochs=5,
        dropout=0.3,
        class_weights=(1.0, 3.0, 3.0, 3.0, 3.0),
        ce_loss_weight=0.5,
        crf_loss_weight=1.0,
        save_path="best_span_roberta_weighted_model.pt"
    )

    model, tokenizer, pos_vocab, ner_vocab, train_ds, val_ds = run_training(
        train_data=train_data,
        val_data=val_data,
        cfg=cfg
    )
    
    print(f"✅ Training complete. Model saved to {cfg.save_path}")

if __name__ == "__main__":
    main()
