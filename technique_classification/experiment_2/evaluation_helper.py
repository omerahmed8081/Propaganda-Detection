# =========================================================
# EVALUATION HELPER FOR SEMEVAL TC TASK
# =========================================================

import os
import subprocess
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# =========================================================
# 1. CREATE GOLD FILE
# =========================================================
def create_gold_file(df, output_path):
    """
    Converts dataframe -> SemEval gold format
    """

    lines = []

    for _, row in df.iterrows():
        article_id = row["article_id"]
        start = int(row["span_start"])
        end = int(row["span_end"])
        techniques = row["techniques"]

        for t in techniques:
            lines.append(f"{article_id}\t{t}\t{start}\t{end}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"✅ Gold file saved to: {output_path}")


# =========================================================
# 2. GENERATE PREDICTIONS FROM MODEL
# =========================================================
@torch.no_grad()
def generate_predictions(
    model,
    dataset,
    tokenizer,
    id2label,
    batch_size=8,
    threshold=0.5,
    device="cuda"
):

    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()

    results = []

    for batch in tqdm(loader, desc="Predicting"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits = model(
            input_ids,
            attention_mask,
            batch["span_start"].to(device),
            batch["span_end"].to(device)
        )

        probs = torch.sigmoid(logits).cpu().numpy()

        for i in range(len(probs)):
            article_id = batch["article_id"][i]
            start = batch["start"][i]
            end = batch["end"][i]

            # 🔥 PREDICT BASED ON CONFIDENCE
            pred_labels = np.where(probs[i] > threshold)[0]

            # 🔥 SAFETY: ensure at least 1 label
            if len(pred_labels) == 0:
                pred_labels = [np.argmax(probs[i])]

            for l in pred_labels:
                results.append(
                    f"{article_id}\t{id2label[l]}\t{start}\t{end}"
                )

    return results

# =========================================================
# 3. SAVE SUBMISSION FILE
# =========================================================
def save_submission(pred_lines, output_path):
    with open(output_path, "w") as f:
        f.write("\n".join(pred_lines))

    print(f"✅ Submission file saved to: {output_path}")


# =========================================================
# 4. RUN SEMEVAL SCORER
# =========================================================
def run_scorer(
    scorer_script_path,
    submission_path,
    gold_path,
    techniques_list_path
):
    """
    Runs your provided scorer script
    """

    command = [
        "python",
        scorer_script_path,
        "-s", submission_path,
        "-r", gold_path,
        "-p", techniques_list_path
    ]

    print("\n🚀 Running SemEval scorer...\n")

    result = subprocess.run(command, capture_output=True, text=True)

    print(result.stdout)

    if result.stderr:
        print("⚠️ STDERR:\n", result.stderr)


# =========================================================
# 5. FULL PIPELINE FUNCTION
# =========================================================
def evaluate_full_pipeline(
    model,
    test_df,
    dataset_class,
    tokenizer,
    label2id,
    id2label,
    scorer_script_path,
    techniques_list_path,
    output_dir="evaluation_outputs",
    batch_size=8,
    threshold=0.5,
    device="cuda"
):

    os.makedirs(output_dir, exist_ok=True)

    gold_path = os.path.join(output_dir, "gold.tsv")
    submission_path = os.path.join(output_dir, "submission.tsv")

    # 1. create gold
    create_gold_file(test_df, gold_path)

    # 2. dataset
    test_dataset = dataset_class(test_df, tokenizer, label2id)

    # 3. predictions
    pred_lines = generate_predictions(
        model,
        test_dataset,
        tokenizer,
        id2label,
        batch_size=batch_size,
        device=device
    )

    # 4. save submission
    save_submission(pred_lines, submission_path)

    # 5. run scorer
    run_scorer(
        scorer_script_path,
        submission_path,
        gold_path,
        techniques_list_path
    )