"""
Few-shot GPT-5.4 propaganda technique classification.

Usage:
    python llm_technique.py                       # run on entire dataset
    python llm_technique.py --num 100             # run on first 100 spans
    python llm_technique.py --num 500 --output_dir ./my_results
"""

import argparse
import json
import os
import random
import re
import time

import pandas as pd
import requests

# =========================================================
# CONFIG  —  fill in your key before running
# =========================================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = "gpt-5.4"

# Directory that contains this script (used for relative defaults)
_HERE = os.path.dirname(os.path.abspath(__file__))

# All 14 propaganda techniques in this dataset
ALL_TECHNIQUES = [
    "Appeal_to_Authority",
    "Appeal_to_fear-prejudice",
    "Bandwagon,Reductio_ad_hitlerum",
    "Black-and-White_Fallacy",
    "Causal_Oversimplification",
    "Doubt",
    "Exaggeration,Minimisation",
    "Flag-Waving",
    "Loaded_Language",
    "Name_Calling,Labeling",
    "Repetition",
    "Slogans",
    "Thought-terminating_Cliches",
    "Whataboutism,Straw_Men,Red_Herring",
]

TECHNIQUE_DEFINITIONS = {
    "Appeal_to_Authority": (
        "Citing an authority figure or institution to support a claim, "
        "without real evidence that the authority is reliable on this topic."
    ),
    "Appeal_to_fear-prejudice": (
        "Using fear, stereotypes, or prejudice to push the audience toward "
        "a conclusion, exploiting emotional vulnerability."
    ),
    "Bandwagon,Reductio_ad_hitlerum": (
        "Bandwagon: persuading by claiming 'everyone is doing it'. "
        "Reductio ad Hitlerum: discrediting an idea by associating it with "
        "Hitler or Nazis."
    ),
    "Black-and-White_Fallacy": (
        "Presenting only two options as if they are the only possibilities, "
        "ignoring nuance or middle ground."
    ),
    "Causal_Oversimplification": (
        "Reducing a complex problem to a single cause, ignoring other "
        "contributing factors."
    ),
    "Doubt": (
        "Questioning the credibility of someone or something without "
        "providing evidence, to plant uncertainty."
    ),
    "Exaggeration,Minimisation": (
        "Exaggeration: overstating the importance of something. "
        "Minimisation: downplaying the importance of something."
    ),
    "Flag-Waving": (
        "Appealing to national, group, or institutional pride to justify "
        "or promote an action or idea."
    ),
    "Loaded_Language": (
        "Using words with strong emotional connotations to influence the "
        "audience beyond what the literal meaning conveys."
    ),
    "Name_Calling,Labeling": (
        "Attaching a negative label or name to a person, group, or idea "
        "to discredit it without substantive argument."
    ),
    "Repetition": (
        "Repeating a word, phrase, or claim so that it becomes accepted "
        "through sheer familiarity."
    ),
    "Slogans": (
        "A brief, memorable phrase used to promote a cause or idea, often "
        "oversimplifying a complex issue."
    ),
    "Thought-terminating_Cliches": (
        "A cliché or platitude used to shut down critical thinking and "
        "end discussion."
    ),
    "Whataboutism,Straw_Men,Red_Herring": (
        "Whataboutism: deflecting criticism by pointing to someone else's flaws. "
        "Straw Man: misrepresenting an argument to make it easier to attack. "
        "Red Herring: introducing an irrelevant topic to distract from the issue."
    ),
}

# =========================================================
# DATA LOADING
# =========================================================
def load_data(data_file: str) -> pd.DataFrame:
    df = pd.read_parquet(data_file)
    df["techniques_list"] = df["techniques"].apply(
        lambda x: list(x) if hasattr(x, "__iter__") and not isinstance(x, str) else [x]
    )
    return df


# =========================================================
# FEW-SHOT EXAMPLE SELECTION
# =========================================================
def get_article_context(article_text: str, span_text: str, window: int = 250) -> str:
    """Return a ~500-char window of the article centred on the span."""
    idx = article_text.find(span_text)
    if idx == -1:
        return article_text[:500]
    start = max(0, idx - window)
    end = min(len(article_text), idx + len(span_text) + window)
    ctx = article_text[start:end]
    if start > 0:
        ctx = "..." + ctx
    if end < len(article_text):
        ctx = ctx + "..."
    return ctx


def pick_examples_per_technique(df: pd.DataFrame) -> dict:
    """
    Pick one real datapoint per technique from the dataset.
    Returns {technique: {span_text, techniques, context}}.
    """
    examples = {}
    for tech in ALL_TECHNIQUES:
        subset = df[df["techniques_list"].apply(lambda tl: tech in tl)]
        if subset.empty:
            continue
        row = subset.iloc[0]
        examples[tech] = {
            "span_text": row["span_text"],
            "techniques": row["techniques_list"],
            "context": get_article_context(row["article_text"], row["span_text"]),
        }
    return examples


# =========================================================
# PROMPT BUILDER
# =========================================================
def build_system_prompt() -> str:
    tech_block = "\n".join(
        f"  - {name}: {desc}" for name, desc in TECHNIQUE_DEFINITIONS.items()
    )
    valid_list = json.dumps(ALL_TECHNIQUES, indent=4)
    return f"""You are an expert in propaganda detection and rhetorical analysis.

Your task is to classify a given propaganda span into one or more of the following techniques.

=== PROPAGANDA TECHNIQUES ===
{tech_block}

=== VALID OUTPUT LABELS ===
{valid_list}

=== RULES ===
- A span can belong to MORE THAN ONE technique.
- Output ONLY a JSON array of technique names, e.g. ["Loaded_Language"] or ["Slogans", "Flag-Waving"].
- Every technique you output MUST be from the valid list above — exact spelling, including commas.
- Do NOT output any explanation or reasoning. Only the JSON array."""


def build_user_prompt(examples: dict, span_text: str, context: str) -> str:
    shots = ""
    for tech, ex in examples.items():
        shots += f"""
--- Example ({tech}) ---
Article context:
{ex['context']}

Propaganda span: "{ex['span_text']}"

Answer: {json.dumps(ex['techniques'])}
"""
    return f"""{shots}

--- Now classify this span ---
Article context:
{context}

Propaganda span: "{span_text}"

Answer:"""


# =========================================================
# API CALL
# =========================================================
def call_gpt(system_prompt: str, user_prompt: str, max_retries: int = 3):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0.0,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=(10, 60),
            )
            print(f"  Status: {response.status_code}")

            if response.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"  Rate limited — sleeping {wait}s...")
                time.sleep(wait)
                continue

            if response.status_code != 200:
                print(f"  Error {response.status_code}: {response.text[:200]}")
                time.sleep(2)
                continue

            return response.json()["choices"][0]["message"]["content"].strip()

        except requests.exceptions.Timeout:
            print(f"  Timeout on attempt {attempt + 1}. Retrying...")
            time.sleep(2)
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(2)

    print("  Failed after all retries.")
    return None


# =========================================================
# OUTPUT PARSER
# =========================================================
def parse_techniques(raw: str) -> list:
    if raw is None:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [t for t in parsed if t in ALL_TECHNIQUES]
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [t for t in parsed if t in ALL_TECHNIQUES]
        except json.JSONDecodeError:
            pass

    # Last resort: scan for known technique names
    return [t for t in ALL_TECHNIQUES if t in raw]


# =========================================================
# METRICS
# =========================================================
def compute_metrics(results: list) -> dict:
    tp = fp = fn = 0
    for r in results:
        gold = set(r["gold_techniques"])
        pred = set(r["predicted_techniques"])
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


# =========================================================
# MAIN PIPELINE
# =========================================================
def run(data_file: str, num_spans, output_dir: str):
    print(f"Loading data from {data_file}...")
    df = load_data(data_file)
    print(f"Total spans in dataset: {len(df)}")

    examples = pick_examples_per_technique(df)
    print(f"Few-shot examples selected for {len(examples)} techniques.")

    target = (df if num_spans is None else df.head(num_spans)).reset_index(drop=True)
    print(f"Classifying {len(target)} spans...\n")

    system_prompt = build_system_prompt()
    results = []

    for idx, row in target.iterrows():
        span_text       = row["span_text"]
        gold_techniques = row["techniques_list"]
        context         = get_article_context(row["article_text"], span_text)

        print(f"[{idx + 1}/{len(target)}] span: {repr(span_text[:60])}")
        print(f"  gold: {gold_techniques}")

        user_prompt = build_user_prompt(examples, span_text, context)
        raw         = call_gpt(system_prompt, user_prompt)
        predicted   = parse_techniques(raw)

        print(f"  predicted: {predicted}")
        print(f"  raw: {repr((raw or '')[:100])}")

        results.append({
            "idx":                  idx,
            "article_id":           row["article_id"],
            "span_text":            span_text,
            "gold_techniques":      gold_techniques,
            "predicted_techniques": predicted,
            "raw_output":           raw,
            "exact_match":          set(predicted) == set(gold_techniques),
        })

        time.sleep(random.uniform(0.5, 1.2))

    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "results.csv"),  index=False)
    results_df.to_json(os.path.join(output_dir, "results.json"), orient="records", indent=2)

    metrics = compute_metrics(results)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== RESULTS ===")
    print(f"Spans classified : {len(results)}")
    print(f"Exact match acc  : {sum(r['exact_match'] for r in results) / len(results):.3f}")
    print(f"Micro Precision  : {metrics['precision']:.3f}")
    print(f"Micro Recall     : {metrics['recall']:.3f}")
    print(f"Micro F1         : {metrics['f1']:.3f}")
    print(f"\nResults saved to: {output_dir}/")


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Few-shot GPT-5.4 propaganda technique classification"
    )
    parser.add_argument(
        "--num",
        type=int,
        default=None,
        help="Number of spans to classify (default: entire dataset)",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=os.path.join(_HERE, "merged_data.parquet"),
        help="Path to merged_data.parquet (default: ./merged_data.parquet)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(_HERE, "output"),
        help="Directory to write results and metrics (default: ./output)",
    )
    args = parser.parse_args()

    run(
        data_file=args.data_file,
        num_spans=args.num,
        output_dir=args.output_dir,
    )
