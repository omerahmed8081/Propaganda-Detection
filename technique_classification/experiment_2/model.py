# =========================================================
# Technique Classification - IMPROVED VERSION
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score
from tqdm import tqdm


# =========================================================
# CONFIG
# =========================================================
class CFG:
    model_name = "roberta-large"
    max_length = 512
    batch_size = 8
    epochs = 10
    threshold = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# LABEL PARSER
# =========================================================
def parse_labels(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, list):
        return x
    return []


# =========================================================
# LABEL MAP
# =========================================================
def build_label_map(df):
    all_labels = set()

    for x in df["techniques"]:
        for l in x:
            all_labels.add(l)

    label2id = {l: i for i, l in enumerate(sorted(all_labels))}
    id2label = {i: l for l, i in label2id.items()}

    print("Total labels:", len(label2id))
    return label2id, id2label


# =========================================================
# CLASS WEIGHTS
# =========================================================
def compute_class_weights(df, label2id):
    counts = np.zeros(len(label2id))

    for labels in df["techniques"]:
        for l in labels:
            if l in label2id:
                counts[label2id[l]] += 1

    total = counts.sum()
    weights = total / (counts + 1e-6)
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float)


# =========================================================
# DATASET
# =========================================================
class TechniqueDataset(Dataset):
    def __init__(self, df, tokenizer, label2id):
        self.samples = []
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.num_labels = len(label2id)

        for _, row in df.iterrows():
            text = row["article_text"]
            start = int(row["span_start"])
            end = int(row["span_end"])

            span_text = text[start:end]

            left = max(0, start - 100)
            right = min(len(text), end + 100)
            context = text[left:right]

            marked_text = (
                context[:start-left]
                + " <SPAN> "
                + span_text
                + " </SPAN> "
                + context[end-left:]
            )

            label_vec = np.zeros(self.num_labels, dtype=np.float32)

            for l in row["techniques"]:
                if l in label2id:
                    label_vec[label2id[l]] = 1

            self.samples.append({
                "text": marked_text,
                "label": label_vec,
                "article_id": row["article_id"],   # 🔥 ADD
                "start": start,
                "end": end
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        enc = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=CFG.max_length,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        offsets = enc["offset_mapping"].squeeze().tolist()

        span_start_token = 0
        span_end_token = 0

        for i, (s, e) in enumerate(offsets):
            if s <= item["start"] < e:
                span_start_token = i
            if s < item["end"] <= e:
                span_end_token = i

        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(item["label"], dtype=torch.float),
            "span_start": torch.tensor(span_start_token),
            "span_end": torch.tensor(span_end_token),
            "article_id": item["article_id"],   # 🔥 ADD
            "start": item["start"],             # 🔥 ADD
            "end": item["end"],                 # 🔥 ADD
        }


# =========================================================
# MODEL (UPGRADED)
# =========================================================
class TechniqueClassifier(nn.Module):
    def __init__(self, model_name, num_labels, pos_weight=None):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.3)

        # CLS + span_mean + span_max
        self.classifier = nn.Linear(hidden_size * 3, num_labels)

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, input_ids, attention_mask, span_start, span_end, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        hidden = outputs.last_hidden_state
        cls = hidden[:, 0]

        batch_size = hidden.size(0)

        span_means = []
        span_maxs = []

        for i in range(batch_size):
            start = span_start[i]
            end = span_end[i]

            if end <= start:
                end = start + 1

            span_tokens = hidden[i, start:end]

            span_mean = span_tokens.mean(dim=0)
            span_max = span_tokens.max(dim=0).values

            span_means.append(span_mean)
            span_maxs.append(span_max)

        span_means = torch.stack(span_means)
        span_maxs = torch.stack(span_maxs)

        rep = torch.cat([cls, span_means, span_maxs], dim=1)

        logits = self.classifier(self.dropout(rep))

        if labels is not None:
            return self.loss_fn(logits, labels)

        return logits


# =========================================================
# TRAIN
# =========================================================
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Training"):
        input_ids = batch["input_ids"].to(CFG.device)
        attention_mask = batch["attention_mask"].to(CFG.device)
        labels = batch["labels"].to(CFG.device)

        optimizer.zero_grad()

        loss = model(
            input_ids,
            attention_mask,
            batch["span_start"].to(CFG.device),
            batch["span_end"].to(CFG.device),
            labels
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# =========================================================
# EVAL
# =========================================================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    all_preds = []
    all_labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(CFG.device)
        attention_mask = batch["attention_mask"].to(CFG.device)

        logits = model(
            input_ids,
            attention_mask,
            batch["span_start"].to(CFG.device),
            batch["span_end"].to(CFG.device)
        )

        probs = torch.sigmoid(logits).cpu().numpy()
        labels = batch["labels"].cpu().numpy()

        all_preds.append(probs)
        all_labels.append(labels)

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    preds_bin = (all_preds > CFG.threshold).astype(int)

    return f1_score(all_labels, preds_bin, average="micro", zero_division=0)


# =========================================================
# TRAIN PIPELINE
# =========================================================
def run_training(train_df, val_df):

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    label2id, id2label = build_label_map(train_df)

    pos_weight = compute_class_weights(train_df, label2id).to(CFG.device)

    train_ds = TechniqueDataset(train_df, tokenizer, label2id)
    val_ds = TechniqueDataset(val_df, tokenizer, label2id)

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size)

    model = TechniqueClassifier(
        CFG.model_name,
        len(label2id),
        pos_weight=pos_weight
    ).to(CFG.device)

    # 🔥 Different LR for encoder vs classifier
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 2e-6},
        {"params": model.classifier.parameters(), "lr": 1e-4}
    ])

    best_f1 = 0

    for epoch in range(CFG.epochs):
        loss = train_one_epoch(model, train_loader, optimizer)
        f1 = evaluate(model, val_loader)

        print(f"\nEpoch {epoch+1}")
        print(f"Loss: {loss:.4f} | F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1

            torch.save({
                "model_state_dict": model.state_dict(),
                "label2id": label2id,
                "id2label": id2label
            }, "best_technique_model.pt")

            print("✅ Model saved!")

    return model, tokenizer, label2id, id2label


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    train_data = pd.read_parquet("/home/omer_ahmed/Experiments/Technique_classification/processed_span_data/train.parquet")
    val_data = pd.read_parquet("/home/omer_ahmed/Experiments/Technique_classification/processed_span_data/val.parquet")

    train_data["techniques"] = train_data["techniques"].apply(parse_labels)
    val_data["techniques"] = val_data["techniques"].apply(parse_labels)

    run_training(train_data, val_data)