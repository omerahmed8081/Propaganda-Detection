import random
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from torchcrf import CRF
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
import logging

# =========================================================
# 1. CONFIG
# =========================================================
@dataclass
class CFG:
    model_name: str = "roberta-large"
    max_length: int = 512
    stride: int = 384
    batch_size: int = 2
    lr: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 5
    warmup_ratio: float = 0.1
    num_workers: int = 2
    seed: int = 42

    lstm_hidden: int = 512
    lstm_layers: int = 1

    pos_dim: int = 32
    ner_dim: int = 32
    discourse_dim: int = 16
    discourse_input_dim: int = 11

    dropout: float = 0.2
    save_path: str = "best_span_roberta_pos_ner_discourse_limited.pt"

    # use CRF only
    class_weights: tuple = (1.0, 1.0, 1.0, 1.0, 1.0)
    ce_loss_weight: float = 0.0
    crf_loss_weight: float = 1.0

    # important: any overlap marks positive
    min_token_overlap_ratio: float = 0.0

    # feature scaling
    pos_scale: float = 0.3
    ner_scale: float = 0.3
    discourse_scale: float = 0.3


# =========================================================
# 2. LABELS
# =========================================================
LABEL2ID = {
    "O": 0,
    "B-PROP": 1,
    "I-PROP": 2,
    "E-PROP": 3,
    "S-PROP": 4,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


# =========================================================
# 3. REPRODUCIBILITY
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# 4. ARTICLE RECORDS
# =========================================================
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


# =========================================================
# 5. POS / NER VOCAB
# =========================================================
def build_pos_ner_vocab(article_records, nlp):
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


# =========================================================
# 6. CHAR-LEVEL AUXILIARY MAPS
# =========================================================
def build_char_feature_maps(text, nlp, pos_vocab, ner_vocab):
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


DISCOURSE_MARKERS = {
    "contrast": {"but", "however", "although", "though", "yet", "while", "whereas"},
    "cause": {"because", "since", "therefore", "thus", "hence", "so"},
    "emphasis": {"indeed", "clearly", "obviously", "certainly", "undoubtedly"},
    "conclusion": {"overall", "finally", "ultimately", "in conclusion", "to conclude"},
}


def safe_div(a, b):
    return a / b if b > 0 else 0.0


def build_char_discourse_feature_map(text, nlp):
    n_chars = len(text)
    feats = np.zeros((n_chars, 11), dtype=np.float32)

    if n_chars == 0:
        return feats

    doc = nlp(text)
    sents = list(doc.sents) if doc.has_annotation("SENT_START") else [doc]

    for i in range(n_chars):
        feats[i, 1] = i / max(1, n_chars - 1)

    paragraph_spans = []
    cursor = 0
    blocks = text.split("\n\n")

    for block in blocks:
        if len(block) == 0:
            cursor += 2
            continue
        block_start = text.find(block, cursor)
        if block_start == -1:
            continue
        block_end = block_start + len(block)
        paragraph_spans.append((block_start, block_end))
        cursor = block_end + 2

    if not paragraph_spans:
        paragraph_spans = [(0, n_chars)]

    for p_start, p_end in paragraph_spans:
        plen = max(1, p_end - p_start)
        for i in range(p_start, p_end):
            feats[i, 2] = (i - p_start) / max(1, plen - 1)

    quote_mask = np.zeros(n_chars, dtype=np.float32)
    in_quote = False
    for i, ch in enumerate(text):
        if ch in ['"', "“", "”", "'", "‘", "’"]:
            in_quote = not in_quote
        if in_quote:
            quote_mask[i] = 1.0
    feats[:, 5] = quote_mask

    n_sents = len(sents)
    for sent_idx, sent in enumerate(sents):
        s_start = sent.start_char
        s_end = sent.end_char
        if s_end <= s_start:
            continue

        sent_text = sent.text.strip()
        sent_len = max(1, s_end - s_start)
        sent_lower = sent_text.lower()

        for i in range(s_start, s_end):
            feats[i, 0] = (i - s_start) / max(1, sent_len - 1)

        if sent_idx == 0:
            feats[s_start:s_end, 3] = 1.0
        if sent_idx == n_sents - 1:
            feats[s_start:s_end, 4] = 1.0

        words = sent_text.split()
        capitalized = sum(1 for w in words if len(w) > 0 and w[0].isupper())
        is_title_like = (
            len(words) <= 12 and
            sent_text.count(".") == 0 and
            safe_div(capitalized, max(1, len(words))) > 0.5
        )
        if is_title_like:
            feats[s_start:s_end, 6] = 1.0

        if any(marker in sent_lower for marker in DISCOURSE_MARKERS["contrast"]):
            feats[s_start:s_end, 7] = 1.0
        if any(marker in sent_lower for marker in DISCOURSE_MARKERS["cause"]):
            feats[s_start:s_end, 8] = 1.0
        if any(marker in sent_lower for marker in DISCOURSE_MARKERS["emphasis"]):
            feats[s_start:s_end, 9] = 1.0
        if any(marker in sent_lower for marker in DISCOURSE_MARKERS["conclusion"]):
            feats[s_start:s_end, 10] = 1.0

    return feats


# =========================================================
# 7. GOLD CHAR MASK
# =========================================================
def build_gold_char_mask(text_len, spans):
    mask = np.zeros(text_len, dtype=np.int64)
    for s, e in spans:
        s = max(0, s)
        e = min(text_len, e)
        if s < e:
            mask[s:e] = 1
    return mask


# =========================================================
# 8. TOKEN TAGGING FROM CHAR SPANS
# any overlap => positive
# =========================================================
def assign_bioes_from_offsets(offsets, gold_char_mask, min_overlap_ratio=0.0):
    token_is_prop = []
    crf_mask = []

    for start, end in offsets:
        if end <= start:
            token_is_prop.append(0)
            crf_mask.append(0)
        else:
            if min_overlap_ratio <= 0.0:
                overlap = gold_char_mask[start:end].max() > 0
                token_is_prop.append(1 if overlap else 0)
            else:
                token_len = max(1, end - start)
                overlap_chars = gold_char_mask[start:end].sum()
                overlap_ratio = overlap_chars / token_len
                token_is_prop.append(1 if overlap_ratio >= min_overlap_ratio else 0)

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


# =========================================================
# 9. DATASET
# =========================================================
class PTCSpanDataset(Dataset):
    def __init__(
        self,
        article_records,
        tokenizer,
        nlp,
        pos_vocab,
        ner_vocab,
        max_length=512,
        stride=384,
        is_train=True,
        min_token_overlap_ratio=0.0
    ):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.is_train = is_train
        self.min_token_overlap_ratio = min_token_overlap_ratio

        for rec in tqdm(article_records, desc="Preparing dataset"):
            article_id = rec["article_id"]
            text = rec["article_text"]
            gold_spans = rec["gold_spans"]

            gold_char_mask = build_gold_char_mask(len(text), gold_spans)

            if nlp is not None:
                pos_char_ids, ner_char_ids = build_char_feature_maps(text, nlp, pos_vocab, ner_vocab)
                discourse_char_feats = build_char_discourse_feature_map(text, nlp)
            else:
                pos_char_ids = np.zeros(len(text), dtype=np.int64)
                ner_char_ids = np.zeros(len(text), dtype=np.int64)
                discourse_char_feats = np.zeros((len(text), 11), dtype=np.float32)

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

                labels, crf_mask = assign_bioes_from_offsets(
                    offsets,
                    gold_char_mask,
                    min_overlap_ratio=self.min_token_overlap_ratio
                )

                pos_ids = []
                ner_ids = []
                discourse_feats = []

                for start, end in offsets:
                    if end <= start:
                        pos_ids.append(0)
                        ner_ids.append(0)
                        discourse_feats.append(np.zeros(11, dtype=np.float32))
                    else:
                        pos_ids.append(int(pos_char_ids[start]))
                        ner_ids.append(int(ner_char_ids[start]))
                        tok_disc = discourse_char_feats[start:end].mean(axis=0)
                        discourse_feats.append(tok_disc.astype(np.float32))

                self.examples.append({
                    "article_id": article_id,
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "crf_mask": torch.tensor(crf_mask, dtype=torch.bool),
                    "pos_ids": torch.tensor(pos_ids, dtype=torch.long),
                    "ner_ids": torch.tensor(ner_ids, dtype=torch.long),
                    "discourse_feats": torch.tensor(np.stack(discourse_feats), dtype=torch.float),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "offset_mapping": offsets,
                    "text_len": len(text),
                    "article_text": text,
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    return {
        "article_id": [x["article_id"] for x in batch],
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "crf_mask": torch.stack([x["crf_mask"] for x in batch]),
        "pos_ids": torch.stack([x["pos_ids"] for x in batch]),
        "ner_ids": torch.stack([x["ner_ids"] for x in batch]),
        "discourse_feats": torch.stack([x["discourse_feats"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
        "offset_mapping": [x["offset_mapping"] for x in batch],
        "text_len": [x["text_len"] for x in batch],
        "article_text": [x["article_text"] for x in batch],
    }


# =========================================================
# 10. MODEL
# kept features, but safer integration
# =========================================================
class TransformerSpanTagger(nn.Module):
    def __init__(
        self,
        model_name="roberta-large",
        num_labels=5,
        num_pos_tags=50,
        num_ner_tags=30,
        pos_dim=32,
        ner_dim=32,
        discourse_input_dim=11,
        discourse_dim=16,
        lstm_hidden=512,
        lstm_layers=1,
        dropout=0.2,
        class_weights=None,
        ce_loss_weight=0.0,
        crf_loss_weight=1.0,
        pos_scale=0.3,
        ner_scale=0.3,
        discourse_scale=0.3
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.pos_dim = pos_dim
        self.ner_dim = ner_dim
        self.discourse_dim = discourse_dim

        self.pos_scale = pos_scale
        self.ner_scale = ner_scale
        self.discourse_scale = discourse_scale

        self.pos_embedding = nn.Embedding(num_pos_tags, pos_dim) if pos_dim > 0 else None
        self.ner_embedding = nn.Embedding(num_ner_tags, ner_dim) if ner_dim > 0 else None

        if discourse_dim > 0:
            self.discourse_proj = nn.Sequential(
                nn.Linear(discourse_input_dim, discourse_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.discourse_proj = None

        fusion_input_dim = hidden_size + pos_dim + ner_dim + discourse_dim
        self.fusion = nn.Linear(fusion_input_dim, hidden_size)
        self.fusion_norm = nn.LayerNorm(hidden_size)

        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden // 2,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

        nn.init.xavier_uniform_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

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

        feats = [token_embeddings]

        if self.pos_embedding is not None:
            feats.append(self.pos_scale * self.pos_embedding(pos_ids))

        if self.ner_embedding is not None:
            feats.append(self.ner_scale * self.ner_embedding(ner_ids))

        if self.discourse_proj is not None and discourse_feats is not None:
            feats.append(self.discourse_scale * self.discourse_proj(discourse_feats))

        fused = torch.cat(feats, dim=-1)
        fused = self.fusion(fused)
        fused = self.fusion_norm(fused)

        lstm_out, _ = self.bilstm(fused)
        lstm_out = self.dropout(lstm_out)
        emissions = self.classifier(lstm_out)

        if labels is not None:
            crf_loss = -self.crf(emissions, labels, mask=crf_mask, reduction="token_mean")

            if self.ce_loss_weight > 0:
                valid_mask = crf_mask.view(-1)
                flat_emissions = emissions.view(-1, emissions.size(-1))[valid_mask]
                flat_labels = labels.view(-1)[valid_mask]
                ce_loss_fct = nn.CrossEntropyLoss(weight=self.class_weights_tensor)
                ce_loss = ce_loss_fct(flat_emissions, flat_labels)
                return self.crf_loss_weight * crf_loss + self.ce_loss_weight * ce_loss

            return self.crf_loss_weight * crf_loss

        return self.crf.decode(emissions, mask=crf_mask)


# =========================================================
# 11. METRICS / SPAN HELPERS
# =========================================================
def token_level_f1(gold_list, pred_list):
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


def _span_overlap_len(span_a, span_b):
    a_start, a_end = span_a
    b_start, b_end = span_b
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _span_len(span):
    return max(0, span[1] - span[0])


def _precision_contribution(pred_span, gold_spans):
    pred_len = _span_len(pred_span)
    if pred_len == 0:
        return 0.0

    best = 0.0
    for gold_span in gold_spans:
        overlap = _span_overlap_len(pred_span, gold_span)
        credit = overlap / pred_len
        if credit > best:
            best = credit
    return best


def _recall_contribution(gold_span, pred_spans):
    gold_len = _span_len(gold_span)
    if gold_len == 0:
        return 0.0

    best = 0.0
    for pred_span in pred_spans:
        overlap = _span_overlap_len(gold_span, pred_span)
        credit = overlap / gold_len
        if credit > best:
            best = credit
    return best


def official_partial_overlap_f1(gold_spans_by_article, pred_spans_by_article):
    all_article_ids = set(gold_spans_by_article.keys()) | set(pred_spans_by_article.keys())

    precision_num = 0.0
    precision_den = 0
    recall_num = 0.0
    recall_den = 0

    for article_id in all_article_ids:
        gold_spans = gold_spans_by_article.get(article_id, [])
        pred_spans = pred_spans_by_article.get(article_id, [])

        for pred_span in pred_spans:
            precision_num += _precision_contribution(pred_span, gold_spans)
            precision_den += 1

        for gold_span in gold_spans:
            recall_num += _recall_contribution(gold_span, pred_spans)
            recall_den += 1

    precision = precision_num / precision_den if precision_den > 0 else 0.0
    recall = recall_num / recall_den if recall_den > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def decode_bioes_token_spans(tag_seq, offsets):
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
            spans.append(offsets[i])
            i += 1
            continue

        if tag == LABEL2ID["B-PROP"]:
            start = offsets[i][0]
            j = i + 1

            if j < n and offsets[j][1] > offsets[j][0] and tag_seq[j] == LABEL2ID["E-PROP"]:
                spans.append((start, offsets[j][1]))
                i = j + 1
                continue

            found_end = False
            while j < n:
                if offsets[j][1] <= offsets[j][0]:
                    break
                if tag_seq[j] == LABEL2ID["I-PROP"]:
                    j += 1
                    continue
                if tag_seq[j] == LABEL2ID["E-PROP"]:
                    spans.append((start, offsets[j][1]))
                    i = j + 1
                    found_end = True
                    break
                break

            if not found_end:
                i += 1
            continue

        i += 1

    return spans


def merge_overlapping_spans(spans):
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


# =========================================================
# 12. PREDICTION
# direct decode + merge only
# =========================================================
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
        discourse_feats = batch["discourse_feats"].to(device)
        labels = batch["labels"].cpu().numpy()

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
    return {rec["article_id"]: rec["gold_spans"] for rec in article_records}


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        crf_mask = batch["crf_mask"].to(device)
        pos_ids = batch["pos_ids"].to(device)
        ner_ids = batch["ner_ids"].to(device)
        discourse_feats = batch["discourse_feats"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            crf_mask=crf_mask,
            pos_ids=pos_ids,
            ner_ids=ner_ids,
            discourse_feats=discourse_feats,
            labels=labels
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


def evaluate(model, loader, article_records, device):
    gold_spans_by_article = build_gold_spans_by_article(article_records)
    pred_spans_by_article, (tok_p, tok_r, tok_f1) = predict_spans(model, loader, device)

    span_p, span_r, span_f1 = exact_span_f1(gold_spans_by_article, pred_spans_by_article)
    official_metrics = official_partial_overlap_f1(gold_spans_by_article, pred_spans_by_article)

    metrics = {
        "token_precision": tok_p,
        "token_recall": tok_r,
        "token_f1": tok_f1,
        "span_precision": span_p,
        "span_recall": span_r,
        "span_f1": span_f1,
        "official_span_precision": official_metrics["precision"],
        "official_span_recall": official_metrics["recall"],
        "official_span_f1": official_metrics["f1"],
    }
    return metrics, pred_spans_by_article


# =========================================================
# 13. TRAINING
# =========================================================
def run_training(train_data, val_data, test_data, cfg: CFG):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_articles = build_article_records(train_data)
    val_articles = build_article_records(val_data)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    if cfg.pos_dim > 0 or cfg.ner_dim > 0 or cfg.discourse_dim > 0:
        nlp = spacy.load("en_core_web_sm", disable=["lemmatizer", "textcat"])
        if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        pos_vocab, ner_vocab = build_pos_ner_vocab(train_articles, nlp)
    else:
        nlp = None
        pos_vocab = {"<PAD>": 0}
        ner_vocab = {"<PAD>": 0}

    train_ds = PTCSpanDataset(
        train_articles, tokenizer, nlp, pos_vocab, ner_vocab,
        max_length=cfg.max_length,
        stride=cfg.stride,
        is_train=True,
        min_token_overlap_ratio=cfg.min_token_overlap_ratio
    )
    val_ds = PTCSpanDataset(
        val_articles, tokenizer, nlp, pos_vocab, ner_vocab,
        max_length=cfg.max_length,
        stride=cfg.stride,
        is_train=False,
        min_token_overlap_ratio=cfg.min_token_overlap_ratio
    )


    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn
    )

    model = TransformerSpanTagger(
        model_name=cfg.model_name,
        num_labels=NUM_LABELS,
        num_pos_tags=len(pos_vocab),
        num_ner_tags=len(ner_vocab),
        pos_dim=cfg.pos_dim,
        ner_dim=cfg.ner_dim,
        discourse_input_dim=cfg.discourse_input_dim,
        discourse_dim=cfg.discourse_dim,
        lstm_hidden=cfg.lstm_hidden,
        lstm_layers=cfg.lstm_layers,
        dropout=cfg.dropout,
        class_weights=cfg.class_weights,
        ce_loss_weight=cfg.ce_loss_weight,
        crf_loss_weight=cfg.crf_loss_weight,
        pos_scale=cfg.pos_scale,
        ner_scale=cfg.ner_scale,
        discourse_scale=cfg.discourse_scale
    ).to(device)

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

    best_val_official_f1 = -1.0

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_metrics, _ = evaluate(model, val_loader, val_articles, device)

        print(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_token_f1={val_metrics['token_f1']:.4f} | "
            f"val_exact_span_f1={val_metrics['span_f1']:.4f} | "
            f"val_official_span_f1={val_metrics['official_span_f1']:.4f}"
        )

        if val_metrics["official_span_f1"] > best_val_official_f1:
            best_val_official_f1 = val_metrics["official_span_f1"]
            best_state = {
                "model_state_dict": model.state_dict(),
                "pos_vocab": pos_vocab,
                "ner_vocab": ner_vocab,
                "cfg": cfg.__dict__,
                "best_val_official_span_f1": best_val_official_f1,
            }
            torch.save(best_state, cfg.save_path)
            print(f"Saved best model to {cfg.save_path}")

    return model, tokenizer, pos_vocab, ner_vocab, train_ds, val_ds


def main():
    # 1. Setup Logging (Optional but recommended for scripts)
    logging.basicConfig(level=logging.INFO)
    print("🚀 Starting Training Pipeline...")

    # 2. Load Data
    train_data = pd.read_parquet("/home/omer_ahmed/Experiments/Span-models/processed_span_data/train.parquet")
    val_data = pd.read_parquet("/home/omer_ahmed/Experiments/Span-models/processed_span_data/val.parquet")
    train_data.drop(columns=["span_text"], inplace=True)
    val_data.drop(columns=["span_text"], inplace=True)

    cfg = CFG(
        model_name="roberta-large",
        stride=384,
        lr=2e-5,
        epochs=5,
        dropout=0.2,
        min_token_overlap_ratio=0.0,
        ce_loss_weight=0.0,
        crf_loss_weight=1.0,
        pos_dim=32,
        ner_dim=32,
        discourse_dim=16,
        pos_scale=0.3,
        ner_scale=0.3,
        discourse_scale=0.3,
    )
    model, tokenizer, pos_vocab, ner_vocab, train_ds, val_ds = run_training(
        train_data=train_data,
        val_data=val_data,
        cfg=cfg
    )

    print(f"✅ Training complete. Model saved to {cfg.save_path}")

if __name__ == "__main__":
    main()