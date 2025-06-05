import argparse
import numpy as np
from collections import Counter

from src.data_loader import load_dna_with_window
from src.regex_generator import load_regex_patterns
from src.model_trainer import extract_features, extract_features_full, extract_one_hot_features, evaluate
from src.decision_tree.model import DecisionTree

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


def parse_args():
    parser = argparse.ArgumentParser(description="Run DNA classifier using a selected decision tree implementation.")

    parser.add_argument("--data_type", choices=["donor", "acceptor"], required=True,
                        help="Type of input data: 'donor' or 'acceptor'")
    parser.add_argument("--data_path",
                        help="Path to the DNA data file")
    parser.add_argument("--regex_path", default="input_data/regex_patterns.txt",
                        help="Path to the regex patterns file")
    parser.add_argument("--window_size", type=int, default=3,
                        help="Window size for extracting DNA sequence")
    parser.add_argument("--max_depth", type=int, default=25,
                        help="Maximum depth of the decision tree")
    parser.add_argument("--min_samples", type=int, default=2,
                        help="Minimum number of samples to split (for custom implementation)")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data used for testing (between 0 and 1)")
    parser.add_argument("--random_state", type=int, default=12,
                        help="Random seed for reproducibility")
    parser.add_argument("--impl", choices=["custom", "sklearn"], default="custom",
                        help="Decision tree implementation to use: 'custom' or 'sklearn'")
    parser.add_argument("--feature_type", choices=["regex", "onehot"], default="regex",
                        help="Feature type to use: 'regex' (default) or 'onehot'")
    parser.add_argument("--regex_search", choices=["full", "window"], default="window",
                        help="Feature type to use: 'full' (default) or 'window'")


    return parser.parse_args()


def main():
    args = parse_args()

    regex_paths = {
        "donor": "input_data/regex_donor.txt",
        "acceptor": "input_data/regex_acceptor.txt"
    }
    data_paths = {
        "donor": "input_data/spliceDTrainKIS.dat",
        "acceptor": "input_data/spliceATrainKIS.dat"
    }

    # 1. Load DNA data
    regexes = load_regex_patterns(regex_paths[args.data_type])
    examples = load_dna_with_window(data_paths[args.data_type], args.data_type)

    # 2. Extract features
    if args.feature_type == "regex":
        if args.regex_search == "full":
            X, y = extract_features_full(examples, regexes)
        elif args.regex_search == "window":
            X, y = extract_features(examples, regexes)
        feature_names = [f"x{i}" for i in range(X.shape[1])]
    elif args.feature_type == "onehot":
        X, y, feature_names = extract_one_hot_features(examples)

    print("Class distribution:", Counter(y))

    # 3. Split into train/val/test
    X_trval, X_test, y_trval, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trval, y_trval, test_size=args.test_size, random_state=args.random_state, stratify=y_trval
    )

    # 4. Select model implementation
    if args.impl == "custom":
        print("\n=== Using: Custom DecisionTree ===")
        model = DecisionTree(
            max_depth=args.max_depth,
            min_samples=args.min_samples,
            n_feats=X.shape[1],
            random_state=args.random_state
        )
    else:
        print("\n=== Using: sklearn DecisionTreeClassifier ===")
        model = DecisionTreeClassifier(
            max_depth=args.max_depth,
            random_state=args.random_state
        )

    # 5. Train and evaluate
    model.fit(X_train, y_train)
    print("\nTest performance:")
    evaluate(model, X_test, y_test)

    from sklearn.metrics import precision_score, recall_score, f1_score

    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    with open("output/metrics_log.tsv", "a") as f:
        f.write(f"{recall:.4f}\t{precision:.4f}\t{f1:.4f}\n")

    # 6. Visualize
    if args.impl == "sklearn":
        plt.figure(figsize=(20, 10))
        plot_tree(model, filled=True, feature_names=feature_names, class_names=["0", "1"])
        plt.title("Decision Tree Visualization (sklearn)")
        plt.savefig("output/decision_tree_sklearn.png")
        plt.show()
    elif args.impl == "custom":
        print("\nText illustration of decision tree (custom):")
        #print(model.export_text(feature_names=feature_names))


if __name__ == "__main__":
    main()
