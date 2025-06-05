import argparse
import numpy as np
import pandas as pd
from collections import Counter
from itertools import product

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import load_dna_with_window
from src.regex_generator import load_regex_patterns
from src.model_trainer import extract_features, extract_features_full, extract_one_hot_features
from src.decision_tree.model import DecisionTree

def tune_hyperparams_crossval(
    data_path: str,
    data_label: str,
    regex_path: str,
    max_depths: list[int],
    min_samples_list: list[int],
    n_splits: int = 5,
    random_state: int = 42,
    feature_type: str = "regex",
    regex_search: str = "window"
):
    print(f"\nTuning for {data_label.upper()} with {n_splits}-fold CV")

    regexes = load_regex_patterns(regex_path)
    examples = load_dna_with_window(data_path, data_label)

    if feature_type == "regex":
        if regex_search == "full":
            X, y = extract_features_full(examples, regexes)
        elif regex_search == "window":
            X, y = extract_features(examples, regexes)
        else:
            raise ValueError("Invalid regex_search mode")
    elif feature_type == "onehot":
        X, y, _ = extract_one_hot_features(examples)
    else:
        raise ValueError("Invalid feature_type")

    results = []
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for md, ms in product(max_depths, min_samples_list):
        scores = []
        for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf = DecisionTree(
                max_depth=md,
                min_samples=ms,
                n_feats=int(np.sqrt(X.shape[1])),
                random_state=random_state + fold
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            scores.append((acc, prec, rec, f1))

        scores = np.array(scores)
        mean_scores = scores.mean(axis=0)
        results.append((md, ms, *mean_scores))
        print(results)

    results.sort(key=lambda x: (x[4], x[2]), reverse=True)
    best = results[0]
    print(f"\nBEST: max_depth={best[0]}, min_samples={best[1]}, Recall={best[4]:.3f}, F1={best[5]:.3f}")

    df = pd.DataFrame(results, columns=["max_depth", "min_samples", "accuracy", "precision", "recall", "f1_score"])
    print("\nFull Results:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    type = ["donor", "acceptor"]
    file = ["input_data/spliceDTrainKIS.dat", "input_data/spliceATrainKIS.dat"]
    regex_paths = ["input_data/regex_donor.txt", "input_data/regex_acceptor.txt"]

    for i, j, k in zip(type, file, regex_paths):
        tune_hyperparams_crossval(
            data_path=j,
            data_label=i,
            regex_path=k,
            max_depths=[3, 5, 10, 15, 20, 30],
            min_samples_list=[2, 5, 10, 20, 30],
            n_splits=10,
            random_state=12,
            feature_type="regex",
            regex_search="window"
        )
