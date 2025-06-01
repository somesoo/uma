import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--features_path', default='features.csv')
parser.add_argument('--regex_input', default='input_data/regex_acceptor.txt')
parser.add_argument('--regex_output', default='input_data/regex_acceptor.txt')
parser.add_argument('--remove_k', type=int, default=5)
args = parser.parse_args()

df = pd.read_csv(args.features_path)

if 'label' not in df.columns:
    print("Brakuje kolumny 'label' w features.csv – nie można policzyć skuteczności.")
    exit(1)

# Wczytaj aktualny plik regexów
with open(args.regex_input, 'r') as f:
    current_regexes = [line.strip() for line in f if line.strip()]

regex_stats = {}

for regex_col in df.columns:
    if regex_col == 'label':
        continue

    pattern = regex_col.split('_', maxsplit=2)[-1]
    if pattern not in current_regexes:
        continue

    TP = ((df[regex_col] == 1) & (df['label'] == 1)).sum()
    FP = ((df[regex_col] == 1) & (df['label'] == 0)).sum()
    FN = ((df[regex_col] == 0) & (df['label'] == 1)).sum()
    TN = ((df[regex_col] == 0) & (df['label'] == 0)).sum()

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    match_1 = TP
    match_0 = FP
    total = match_1 + match_0
    class1_ratio = match_1 / (total + 1e-6)

    # ambiwalentność: blisko 0.5 = źle, blisko 0 lub 1 = dobrze
    ambiguity = 1 - abs(class1_ratio - 0.5) * 2

    # końcowy score: im wyżej, tym gorzej
    score = (1 - f1) + ambiguity

    regex_stats[pattern] = score

# Posortuj regexy według oceny
sorted_regexes = sorted(regex_stats.items(), key=lambda x: x[1])
to_remove = [r for r, _ in sorted_regexes[:args.remove_k]]
updated_regexes = [r for r in current_regexes if r not in to_remove]

# Zapisz nowy plik
with open(args.regex_output, 'w') as f:
    for r in updated_regexes:
        f.write(r + '\n')

print(f"Usunięto {len(to_remove)} regexów o słabym F1 lub ambiwalentnym dopasowaniu:")
for r in to_remove:
    print(f"  - {r}")