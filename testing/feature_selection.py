import itertools
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from src.data_loader import load_dna_with_window
from src.regex_generator import load_regex_patterns
from src.model_trainer import extract_features
from src.decision_tree.model import DecisionTree

def evaluate(y_true, y_pred):
    return [
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        f1_score(y_true, y_pred, zero_division=0)
    ]

def feature_selection_search(
    data_path: str,
    data_label: str,
    regex_path: str,
    window_size: int,
    all_positions: list[int],
    max_depth: int = 10,
    min_samples: int = 2,
    test_size: float = 0.2,
    random_state: int = 42,
    n_repeats: int = 5
):
    examples = load_dna_with_window(data_path, data_label, window_size)
    regexes = load_regex_patterns(regex_path)

    position_pairs = list(itertools.combinations(all_positions, 2))
    results = []

    for pos_pair in position_pairs:
        positions = list(pos_pair)
        X, y = extract_features(examples, regexes, positions)

        metrics_list = []

        for i in range(n_repeats):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state + i, stratify=y
            )

            clf = DecisionTree(
                max_depth=max_depth,
                min_samples=min_samples,
                n_feats=int(X.shape[1]),
                random_state=random_state + i
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            metrics = evaluate(y_test, y_pred)
            metrics_list.append(metrics)

        metrics_array = np.array(metrics_list)
        mean_metrics = metrics_array.mean(axis=0)  # [acc, prec, recall, f1]
        results.append((pos_pair, *mean_metrics))

    results.sort(key=lambda x: (x[3], x[4]), reverse=True)  # sort by recall, then F1

    print("\nBest position pairs (sorted by recall → f1):")
    for (pos1, pos2), acc, prec, recall, f1 in results:
        print(f"Positions: [{pos1}, {pos2}] → Acc={acc:.3f}, Prec={prec:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

    return results

if __name__ == "__main__":
    feature_selection_search(
        data_path="input_data/spliceDTrainKIS.dat",
        data_label="donor",
        regex_path="input_data/regex_patterns.txt",
        window_size=5,
        all_positions=[5, 7, 10, 30, 60, 68],
        max_depth=10,
        min_samples=2,
        n_repeats=5  
    )
