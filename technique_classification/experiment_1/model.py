import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import config

# =========================================================
# Technique Classification - FINAL WORKING VERSION
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
    lr = 2e-5
    epochs = 5
    threshold = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# LABEL PARSER (FINAL FIX)
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
# DATASET
# =========================================================
class TechniqueDataset(Dataset):
    def __init__(self, df, tokenizer, label2id):
        self.samples = []
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.num_labels = len(label2id)

        positive_samples = 0

        for _, row in df.iterrows():
            text = row["article_text"]
            start = int(row["span_start"])
            end = int(row["span_end"])

            span_text = text[start:end]

            # context window
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

            labels = row["techniques"]

            for l in labels:
                if l in label2id:
                    label_vec[label2id[l]] = 1

            if label_vec.sum() > 0:
                positive_samples += 1

            self.samples.append({
                "text": marked_text,
                "label": label_vec,
                "article_id": row["article_id"],
                "start": start,
                "end": end
            })

        print(f"Dataset built: {positive_samples}/{len(self.samples)} positive samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        enc = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=CFG.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(item["label"], dtype=torch.float),
            "article_id": item["article_id"],
            "start": item["start"],
            "end": item["end"]
        }


# =========================================================
# MODEL
# =========================================================
class TechniqueClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        cls = outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(cls))

        if labels is not None:
            loss = self.loss_fn(logits, labels)

            if torch.isnan(loss):
                return torch.tensor(0.0, requires_grad=True).to(logits.device)

            return loss

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

        loss = model(input_ids, attention_mask, labels)

        if torch.isnan(loss):
            continue

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

        labels = batch["labels"].cpu().numpy()

        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_preds.append(probs)
        all_labels.append(labels)

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    preds_bin = (all_preds > CFG.threshold).astype(int)

    if all_labels.sum() == 0:
        return 0.0

    return f1_score(all_labels, preds_bin, average="micro", zero_division=0)


# =========================================================
# TRAIN PIPELINE
# =========================================================
def run_training(train_df, val_df):

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    label2id, id2label = build_label_map(train_df)

    train_ds = TechniqueDataset(train_df, tokenizer, label2id)
    val_ds = TechniqueDataset(val_df, tokenizer, label2id)

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size)

    model = TechniqueClassifier(CFG.model_name, len(label2id)).to(CFG.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)

    best_f1 = 0

    for epoch in range(CFG.epochs):
        loss = train_one_epoch(model, train_loader, optimizer)
        f1 = evaluate(model, val_loader)

        print(f"\nEpoch {epoch+1}")
        print(f"Loss: {loss:.4f} | F1: {f1:.4f}")

        # 🔥 SAVE BEST MODEL
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

    train_data = pd.read_parquet(config.TECHNIQUE_TRAIN_PARQUET)
    val_data = pd.read_parquet(config.TECHNIQUE_VAL_PARQUET)
    test_data = pd.read_parquet(config.TECHNIQUE_TEST_PARQUET)

    # 🔥 CRITICAL FIX (YOU WERE MISSING THIS)
    train_data["techniques"] = train_data["techniques"].apply(parse_labels)
    val_data["techniques"] = val_data["techniques"].apply(parse_labels)
    test_data["techniques"] = test_data["techniques"].apply(parse_labels)

    model, tokenizer, label2id, id2label = run_training(train_data, val_data)