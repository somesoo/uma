import pandas as pd
import re

def pretty_name(col: str) -> str:
    """Zwraca sformatowaną nazwę regexu z numerem i wzorcem"""
    match = re.match(r"rx_(\d+)_(.+)", col)
    if match:
        return f"regex {match.group(1)}:\t{match.group(2)}"
    return col  # fallback

def analyze_feature_matches(csv_path: str, top_n: int = 50) -> None:
    # Wczytaj dane
    df = pd.read_csv(csv_path)
    features = df.drop(columns=["label"])
    labels = df["label"]

    # Policz dopasowania dla obu klas
    matches_1 = features[labels == 1].sum()
    matches_0 = features[labels == 0].sum()
    total_matches = matches_1 + matches_0

    # Top N dla klasy 1
    print(f"\n{top_n} regexów najczęściej dopasowanych dla klasy 1:\n")
    for col in matches_1.sort_values(ascending=False).head(top_n).index:
        print(f"{pretty_name(col)}\t{matches_1[col]}")

    # Top N dla klasy 0
    print(f"\n{top_n} regexów najczęściej dopasowanych dla klasy 0:\n")
    for col in matches_0.sort_values(ascending=False).head(top_n).index:
        print(f"{pretty_name(col)}\t{matches_0[col]}")

    # Top N dla obu klas razem
    print(f"\n{top_n} regexów z największą łączną liczbą dopasowań (klasa 1 + 0):\n")
    for col in total_matches.sort_values(ascending=False).head(top_n).index:
        print(f"{pretty_name(col)}\tklasa 1: {matches_1[col]}\tklasa 0: {matches_0[col]}\trazem: {total_matches[col]}")

if __name__ == "__main__":
    analyze_feature_matches("features.csv")
