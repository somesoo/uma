### Jonatan Kasperczak

import pandas as pd
import re

TOP_K = 100
BATCH_SIZE = 10
SCORES_FILE = "scores.tsv"
POOL_FILE = "regex_pool.txt"
OUTPUT_FILE = "input_data/regex_donor.txt"
STATE_FILE = "regex_selection_state.txt"  # śledzenie pozycji w puli

def canonicalize_regex(r):
    return re.sub(r"[.]+$", "", r)

def load_used_regexes():
    try:
        with open(OUTPUT_FILE) as f:
            return set(line.strip() for line in f)
    except FileNotFoundError:
        return set()

def load_next_batch(pool_path, skip_n):
    with open(pool_path) as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines[skip_n : skip_n + BATCH_SIZE]

def save_selection_state(next_index):
    with open(STATE_FILE, "w") as f:
        f.write(str(next_index))

def load_selection_state():
    try:
        with open(STATE_FILE) as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return 0

def main():
    # 1. Wczytaj wyniki
    df = pd.read_csv(SCORES_FILE, sep="\t")
    total_class1 = 1120  # <- przykładowo, zlicz raz na początku eksperymentu

    df["recall"] = df["matches_class1"] / total_class1
    df["precision"] = df["matches_class1"] / (df["matches_class1"] + df["matches_class0"] + 1e-6)
    df["score"] = df["recall"] + df["precision"]
    df = df[df["matches_class1"] >= 5]
    df = df[df["total"] <= 1099]
    df["pattern"] = df["regex"].apply(lambda r: r.split("_", 2)[-1])
    df = df.drop_duplicates("pattern")

    df = df.sort_values("score", ascending=False)
    top_regexes = df["regex"].head(TOP_K)

    # 2. Przygotuj nowe regexy z puli
    start_index = load_selection_state()
    new_batch = load_next_batch(POOL_FILE, start_index)
    next_index = start_index + BATCH_SIZE
    save_selection_state(next_index)

    print(f"Adding {BATCH_SIZE} new regexes from index {start_index}")

    # 3. Zbuduj nowy zbiór
    selected_patterns = [r.split("_", 2)[-1] for r in top_regexes]
    updated_patterns = selected_patterns + new_batch

    unique_patterns = {}
    for regex in selected_patterns:
        canon = canonicalize_regex(regex)
        if canon not in unique_patterns:
            unique_patterns[canon] = regex
    with open(OUTPUT_FILE, "w") as f:
        for r in unique_patterns.values():
            f.write(r + "\n")

    print(f"Saved {len(updated_patterns)} regexes to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
