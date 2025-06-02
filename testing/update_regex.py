import pandas as pd
import argparse

def compute_regex_score(df, pattern, col_name):
    TP = ((df[col_name] == 1) & (df['label'] == 1)).sum()
    FP = ((df[col_name] == 1) & (df['label'] == 0)).sum()
    FN = ((df[col_name] == 0) & (df['label'] == 1)).sum()

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    total = TP + FP
    class1_ratio = TP / (total + 1e-6)
    ambiguity = 1 - abs(class1_ratio - 0.5) * 2  # 0 dla czystego, 1 dla maks. niepewności

    score = (1 - f1) + ambiguity
    return score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_path', default='features.csv')
    parser.add_argument('--regex_input', default='input_data/regex_acceptor.txt')
    parser.add_argument('--regex_output', default='input_data/regex_acceptor.txt')
    parser.add_argument('--remove_k', type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.features_path)
    if 'label' not in df.columns:
        raise ValueError("Brakuje kolumny 'label' w features.csv – nie można policzyć skuteczności.")

    with open(args.regex_input, 'r') as f:
        current_regexes = [line.strip() for line in f if line.strip()]

    regex_scores = {}
    for col in df.columns:
        if col == 'label':
            continue
        pattern = col.split('_', maxsplit=2)[-1]
        if pattern in current_regexes:
            regex_scores[pattern] = compute_regex_score(df, pattern, col)

    # Usuwamy najsłabsze regexy wg oceny
    sorted_scores = sorted(regex_scores.items(), key=lambda x: x[1])
    to_remove = {r for r, _ in sorted_scores[:args.remove_k]}
    updated_regexes = [r for r in current_regexes if r not in to_remove]

    with open(args.regex_output, 'w') as f:
        for r in updated_regexes:
            f.write(r + '\n')

    print(f"Usunięto {len(to_remove)} regexów o słabym F1 lub ambiwalentnym dopasowaniu:")
    for r in to_remove:
        print(f"  - {r}")

if __name__ == "__main__":
    main()
