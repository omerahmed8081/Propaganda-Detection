import pandas as pd
import requests
import json
import random
import re
import time
import argparse
OPENAI_API_KEY = ""
MODEL = "gpt-5.4"
URL = "https://api.openai.com/v1/chat/completions"

def remove_spans_from_text(text, spans):
    clean_text = text

    for span in spans:
        pattern = re.escape(span)
        clean_text = re.sub(pattern, "", clean_text, count=1)

    # Clean extra spaces
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    return clean_text


def build_examples_from_ids(df, positive_ids, negative_ids, exclude_article=None):

    positive_examples = []
    negative_examples = []

    grouped = df.groupby("article_id")

    # =====================
    # POSITIVE EXAMPLES
    # =====================
    for pid in positive_ids:
        if pid not in grouped.groups:
            continue

        group = grouped.get_group(pid)
        article_text = group["article_text"].iloc[0]

        if exclude_article is not None and article_text == exclude_article:
            continue

        spans = list(set(group["span_text"].dropna().tolist()))

        if len(spans) == 0:
            continue

        positive_examples.append({
            "article_text": article_text,
            "spans": spans
        })

    # =====================
    # NEGATIVE EXAMPLES
    # =====================
    for nid in negative_ids:
        if nid not in grouped.groups:
            continue

        group = grouped.get_group(nid)
        article_text = group["article_text"].iloc[0]

        if exclude_article is not None and article_text == exclude_article:
            continue

        spans = list(set(group["span_text"].dropna().tolist()))

        clean_text = remove_spans_from_text(article_text, spans)

        negative_examples.append({
            "article_text": clean_text
        })

    return positive_examples, negative_examples
# =========================
# 2. BUILD PROMPT (IMPROVED)
# =========================
def build_prompt(positive_examples, negative_examples, new_article):

    system_prompt = """You are an expert in propaganda detection.

    Propaganda is language designed to manipulate opinions using:
    - emotional appeal
    - exaggeration
    - bias
    - fear
    - political persuasion

    You MUST follow this reasoning process internally:
    1. Read the article carefully
    2. Identify emotionally charged or manipative phrases
    3. Ignore neutral factual statements
    4. Extract ONLY exact spans from the text

    IMPORTANT:
    - Do NOT output your reasoning
    - Output ONLY final JSON list
    """

    # =====================
    # POSITIVE EXAMPLES
    # =====================
    pos_text = ""
    for i, ex in enumerate(positive_examples):
        pos_text += f"""
    Positive Example {i+1}:
    Article:
    {ex['article_text']}

    Reasoning:
    - Identify propaganda spans based on emotional language, exaggeration, bias, etc.

Answer:
{json.dumps(ex['spans'], ensure_ascii=False)}
"""

    # =====================
    # NEGATIVE EXAMPLES
    # =====================
    neg_text = ""
    for i, ex in enumerate(negative_examples):
        neg_text += f"""
Negative Example {i+1}:
Article:
{ex['article_text']}

Reasoning:
- No manipulation or propaganda detected

Answer:
[]
"""

    # =====================
    # FINAL TASK
    # =====================
    user_prompt = f"""
{pos_text}

{neg_text}

Now analyze the following article.

Article:
{new_article}

Follow the same reasoning process internally.

Return ONLY:
["span1", "span2", ...]
"""

    return system_prompt, user_prompt


# =========================
# 3. CALL OPENROUTER
# =========================


def call_openrouter(system_prompt, user_prompt, max_retries=3):

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=(10, 60)
            )

            print(f"Status: {response.status_code}")

            if response.status_code == 429:
                print("Rate limited. Sleeping...")
                time.sleep(5 * (attempt + 1))
                continue

            result = response.json()

            return result["choices"][0]["message"]["content"]

        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt+1}. Retrying...")
            time.sleep(2)

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2)

    print("Failed after retries.")
    return None


# =========================
# 4. SAFE JSON PARSER
# =========================
def parse_output(output):
    try:
        return json.loads(output)
    except:
        # Try to extract JSON manually
        match = re.search(r"\[.*\]", output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                return []
        return []


# =========================
# 5. POST-PROCESS FILTER
# =========================
def filter_spans(spans):
    filtered = []

    for s in spans:
        s = s.strip()

        # Remove short spans
        if len(s.split()) <= 2:
            continue

        # Remove numeric / weak spans
        if s.isdigit():
            continue

        # Deduplicate
        if s not in filtered:
            filtered.append(s)

    return filtered


# =========================
# 6. FULL PIPELINE
# =========================
def predict_spans(df, new_article):

    positive_ids = ["111111112", "111111113", "111111115"]
    negative_ids = ["111111117", "111111123"]

    positive_examples, negative_examples = build_examples_from_ids(
        df,
        positive_ids,
        negative_ids,
        exclude_article=new_article
    )

    system_prompt, user_prompt = build_prompt(
        positive_examples,
        negative_examples,
        new_article
    )

    raw_output = call_openrouter(system_prompt, user_prompt)
    if raw_output is None:
        return {
            "raw_output": None,
            "parsed_spans": [],
            "filtered_spans": []
        }
    parsed = parse_output(raw_output)

    filtered = filter_spans(parsed)

    return {
        "raw_output": raw_output,
        "parsed_spans": parsed,
        "filtered_spans": filtered
    }
import pandas as pd

def predict_on_dataset(train_df, test_df, num_articles=5):

    results = []

    # Get unique articles
    unique_articles = test_df["article_id"].unique()

    if num_articles is not None:
        unique_articles = unique_articles[:num_articles]
    for article_id in unique_articles:

        article_df = test_df[test_df["article_id"] == article_id]

        article_text = article_df["article_text"].iloc[0]
        true_spans = list(set(article_df["span_text"].dropna().tolist()))
        print(f"\nProcessing Article ID: {article_id}")
        try:
            prediction = predict_spans(train_df, article_text)
            time.sleep(random.uniform(0.8, 1.5))
            predicted_spans = prediction["filtered_spans"]

        except Exception as e:
            print(f"Error processing article {article_id}: {e}")
            predicted_spans = []

        results.append({
            "article_id": article_id,
            "article_text": article_text,
            "true_spans": true_spans,
            "predicted_spans": predicted_spans
        })

    return pd.DataFrame(results)
from difflib import SequenceMatcher

def get_best_overlap(a, b):
    matcher = SequenceMatcher(None, a, b)
    match = matcher.find_longest_match(0, len(a), 0, len(b))

    if match.size == 0:
        return ""

    return a[match.a: match.a + match.size]

def build_partial_match_df(results_df):

    rows = []

    for _, row in results_df.iterrows():

        article_id = row["article_id"]
        true_spans = row["true_spans"]
        predicted_spans = row["predicted_spans"]

        for pred in predicted_spans:
            for gold in true_spans:

                # Check partial match
                if pred in gold or gold in pred:

                    overlap = get_best_overlap(pred, gold)

                    rows.append({
                        "article_id": article_id,
                        "gold_span": gold,
                        "predicted_span": pred,
                        "matched_part": overlap
                    })

    return pd.DataFrame(rows)
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run LLM span prediction")
    parser.add_argument(
        "--num_articles",
        type=int,
        default=None,
        help="Number of articles to process (default: all)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "val"],
        default="val",
        help="Dataset to run on: train or val (default: val)"
    )
    args = parser.parse_args()

    # Load data
    data = pd.read_parquet("processed_span_data/data.parquet")
    dev_data = pd.read_parquet("processed_span_data/dev.parquet")

    # Decide number of articles
    if args.num_articles is not None:
        num_articles = args.num_articles
        print(f"Running on {num_articles} articles...")
    else:
        num_articles = len(dev_data["article_id"].unique())
        print(f"Running on ALL articles ({num_articles})...")

    # Decide dataset
    if args.mode == "val":
        test_data = dev_data
        print("Running on VALIDATION dataset")
    elif args.mode == "train":
        test_data = data
        print("Running on TRAIN dataset")
    # Run prediction
    results_df = predict_on_dataset(
        data,          # training data still used for examples
        test_data,     # this changes based on mode
        num_articles=num_articles
    )
    partial_matches_df = build_partial_match_df(results_df)
    results_df.to_csv("results.csv", index=False)
    partial_matches_df.to_csv("partial_matches.csv", index=False)
    results_df.to_json("results.json", orient="records", indent=2)
    print("Done! Results saved to partial_matches.csv")