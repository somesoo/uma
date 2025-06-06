### Jonatan Kasperczak

import pandas as pd
import re

SCORES_FILE = "scores.tsv"
OUTPUT_METRICS_FILE = "best_regexes_metrics.tsv"
TOP_K = 100
TOTAL_CLASS1 = 1120
ALPHA = 1.5
MIN_PRECISION = 0.1
MIN_MATCHES_CLASS1 = 5

def canonicalize_regex(r):
    return re.sub(r"[.]+$", "", r)

def main():
    df = pd.read_csv(SCORES_FILE, sep="\t")

    df["precision"] = df["matches_class1"] / (df["matches_class1"] + df["matches_class0"])
    df["recall"] = df["matches_class1"] / TOTAL_CLASS1
    df["score"] = df["matches_class1"] - ALPHA * df["matches_class0"]

    df = df[df["precision"] >= MIN_PRECISION]
    df = df[df["matches_class1"] >= MIN_MATCHES_CLASS1]

    df["pattern"] = df["regex"].apply(lambda r: r.split("_", 2)[-1])
    df["canonical"] = df["pattern"].apply(canonicalize_regex)
    df["length"] = df["pattern"].str.len()
    df = df.sort_values(by=["canonical", "length"])
    df = df.drop_duplicates("canonical")

    df = df.sort_values("score", ascending=False)

    top_df = df.head(TOP_K)
    top_df[["pattern", "precision", "recall", "score"]].to_csv(OUTPUT_METRICS_FILE, sep="\t", index=False)


if __name__ == "__main__":
    main()