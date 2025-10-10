import re
import pandas as pd
from collections import Counter


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re.sub(r'[^\w\s]', '', text)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_em(pred: str, true: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(true) else 0.0


def compute_f1(pred: str, true: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    true_tokens = normalize_answer(true).split()
    common = Counter(pred_tokens) & Counter(true_tokens)
    num_same = sum(common.values())

    if len(pred_tokens) == 0 or len(true_tokens) == 0:
        return float(pred_tokens == true_tokens)
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(true_tokens)
    return 2 * precision * recall / (precision + recall)


def debug_overlap(pred: str, true: str):
    """Show token-level comparison for debugging."""
    pred_toks = normalize_answer(pred).split()
    true_toks = normalize_answer(true).split()
    common = set(pred_toks) & set(true_toks)
    return {
        "pred_tokens": pred_toks,
        "true_tokens": true_toks,
        "common_tokens": list(common),
    }


if __name__ == "__main__":
    # Load CSVs
    gt_df = pd.read_csv(
        ""
    )
    pred_df = pd.read_csv(
        ""
    )
    pred_df = pred_df.loc[pred_df["mode"] == "graph_of_thought"]

    # Merge on question (ensure both have a 'question' column)
    merged = pd.merge(gt_df, pred_df, on="question", how="inner")

    # Compute metrics row-wise
    merged["EM"] = merged.apply(lambda row: compute_em(
        row["response"], row["answer"]), axis=1)
    merged["F1"] = merged.apply(lambda row: compute_f1(
        row["response"], row["answer"]), axis=1)

    # Report scores
    print(f"Exact Match (EM): {merged['EM'].mean():.4f}")
    print(f"F1 Score: {merged['F1'].mean():.4f}\n")

    # Save results for inspection
    merged.to_csv("evaluation_results.csv", index=False)

    # 🔎 Print all examples with the lowest F1
    min_f1 = merged["F1"].min()
    worst_cases = merged.loc[merged["F1"] == min_f1]

    print("=" * 80)
    print(f"Lowest F1 Score: {min_f1:.4f}")
    print(f"Number of cases with this score: {len(worst_cases)}")
    print("=" * 80)

    for i, row in worst_cases.iterrows():
        print(f"Question: {row['question']}")
        print(f"Ground Truth Answer: {row['answer']}")
        print(f"Model Prediction: {row['response']}")
        print(f"EM: {row['EM']}, F1: {row['F1']}")
        print("Token overlap:", debug_overlap(row["response"], row["answer"]))
        print("-" * 80)
