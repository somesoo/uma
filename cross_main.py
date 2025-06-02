import argparse
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

from src.data_loader import load_dna_with_window
from src.regex_generator import load_regex_patterns
from src.model_trainer import extract_features, extract_features_full, extract_one_hot_features
from src.decision_tree.model import DecisionTree
from sklearn.tree import DecisionTreeClassifier

def parse_args():
    parser = argparse.ArgumentParser(description="Cross-validated DNA classifier using decision trees")

    parser.add_argument("--data_type", choices=["donor", "acceptor"], required=True)
    parser.add_argument("--data_path")
    parser.add_argument("--regex_path", default="input_data/regex_patterns.txt")
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--max_depth", type=int, default=30)
    parser.add_argument("--min_samples", type=int, default=2)
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--impl", choices=["custom", "sklearn"], default="custom")
    parser.add_argument("--feature_type", choices=["regex", "onehot"], default="regex")
    parser.add_argument("--regex_search", choices=["full", "window"], default="window")

    return parser.parse_args()

def main():
    args = parse_args()

    data_paths = {
        "donor": "input_data/spliceDTrainKIS.dat",
        "acceptor": "input_data/spliceATrainKIS.dat"
    }
    regex_paths = {
        "donor": "input_data/regex_donor.txt",
        "acceptor": "input_data/regex_acceptor.txt"
    }

    data_path = data_paths[args.data_type]
    regex_path = regex_paths[args.data_type]

    regexes = load_regex_patterns(regex_path)
    regex_len = len(regexes[0])
    examples = load_dna_with_window(data_path, args.data_type, regex_len)

    if args.feature_type == "regex":
        if args.regex_search == "full":
            X, y = extract_features_full(examples, regexes)
        else:
            X, y = extract_features(examples, regexes)
    else:
        X, y, _ = extract_one_hot_features(examples)

    print("Class distribution:", Counter(y))

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if args.impl == "custom":
            model = DecisionTree(
                max_depth=args.max_depth,
                min_samples=args.min_samples,
                n_feats=X.shape[1],
                random_state=args.random_state
            )
        else:
            model = DecisionTreeClassifier(
                max_depth=args.max_depth,
                random_state=args.random_state
            )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            "fold": fold,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0)
        })

    df = pd.DataFrame(results)
    df.loc["avg"] = df.mean(numeric_only=True)

    print("\n=== Cross-validation results ===")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
