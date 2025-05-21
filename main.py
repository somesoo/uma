import argparse
import numpy as np
from collections import Counter

from src.data_loader import load_dna_with_window
from src.regex_generator import load_regex_patterns
from src.model_trainer import extract_features, evaluate
from src.decision_tree.model import DecisionTree

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Run DNA classifier using a selected decision tree implementation.")

    parser.add_argument("--data_type", choices=["donor", "acceptor"], required=True,
                        help="Type of input data: 'donor' or 'acceptor'")
    parser.add_argument("--data_path", required=True,
                        help="Path to the DNA data file")
    parser.add_argument("--regex_path", default="input_data/regex_patterns.txt",
                        help="Path to the regex patterns file")
    parser.add_argument("--window_size", type=int, default=5,
                        help="Window size for extracting DNA sequence")
    parser.add_argument("--max_depth", type=int, default=10,
                        help="Maximum depth of the decision tree")
    parser.add_argument("--min_samples", type=int, default=2,
                        help="Minimum number of samples to split (for custom implementation)")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data used for testing (between 0 and 1)")
    parser.add_argument("--random_state", type=int, default=48,
                        help="Random seed for reproducibility")
    parser.add_argument("--impl", choices=["custom", "sklearn"], default="custom",
                        help="Decision tree implementation to use: 'custom' or 'sklearn'")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.data_type == "acceptor":
        positions = [68]
    elif args.data_type == "donor":
        positions = [7]
    else:
        raise ValueError(f"Unsupported data_type: {args.data_type}")

    # 1. Load DNA data
    examples = load_dna_with_window(args.data_path, args.data_type)
    regexes = load_regex_patterns(args.regex_path)
    X, y = extract_features(examples, regexes)
    print("Class distribution:", Counter(y))

    # 2. Split into train/val/test
    X_trval, X_test, y_trval, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trval, y_trval, test_size=args.test_size, random_state=args.random_state, stratify=y_trval
    )

    # 3. Select model implementation
    if args.impl == "custom":
        print("\n=== Using: Custom DecisionTree ===")
        model = DecisionTree(
            max_depth=args.max_depth,
            min_samples=args.min_samples,
            n_feats=int(np.sqrt(X.shape[1])),
            random_state=args.random_state
        )
    else:
        print("\n=== Using: sklearn DecisionTreeClassifier ===")
        model = DecisionTreeClassifier(
            max_depth=args.max_depth,
            random_state=args.random_state
        )

    # 4. Train and evaluate
    model.fit(X_train, y_train)
    print("\nValidation performance:")
    evaluate(model, X_val, y_val)
    print("\nTest performance:")
    evaluate(model, X_test, y_test)


if __name__ == "__main__":
    main()
