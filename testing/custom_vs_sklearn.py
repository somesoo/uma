import numpy as np
import time
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.data_loader import load_dna_with_window
from src.regex_generator import load_regex_patterns
from src.model_trainer import extract_features
from src.decision_tree.model import DecisionTree

def evaluate(model, X, y_true, name=""):
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"[{name}] Acc: {acc:.2f}, Prec: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")
    return acc, prec, rec, f1

def compare_models(
    data_label: str,
    window_size: int = 5,
    max_depth: int = 10,
    min_samples: int = 2,
    test_size: float = 0.2,
    random_state: int = 42
):
    print(f"\n=== Comparing models for '{data_label}' dataset ===")

    if data_label.lower() == "acceptor":
        positions = [68]
    elif data_label.lower() == "donor":
        positions = [7]
    else:
        raise ValueError(f"Unknown data_label: {data_label}. Expected 'acceptor' or 'donor'.")
    data_path = {
        "donor": "input_data/spliceDTrainKIS.dat",
        "acceptor": "input_data/spliceATrainKIS.dat"
    }
    regex_paths = {
        "donor": "input_data/regex_donor.txt",
        "acceptor": "input_data/regex_acceptor.txt"
    }  
    regexes = load_regex_patterns(regex_paths[data_label])
    # 1. Load and extract features
    examples = load_dna_with_window(data_path[data_label], data_label, len(regexes[0]))

    X, y     = extract_features(examples, regexes)
    print("Class balance:", Counter(y))

    # 2. Split dataset
    X_trval, X_test, y_trval, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_trval, y_trval, test_size=test_size, random_state=random_state, stratify=y_trval)

    n_feats = int(np.sqrt(X.shape[1]))

    # 3. Custom Tree
    print("\n--- Custom DecisionTree ---")
    start = time.time()
    custom_clf = DecisionTree(max_depth=max_depth, min_samples=min_samples, n_feats=n_feats, random_state=random_state)
    custom_clf.fit(X_train, y_train)
    train_time_custom = time.time() - start
    print(f"Training time: {train_time_custom:.2f} s")
    evaluate(custom_clf, X_val, y_val, "Custom - Val")
    evaluate(custom_clf, X_test, y_test, "Custom - Test")

    # 4. Sklearn Tree
    print("\n--- sklearn DecisionTreeClassifier ---")
    start = time.time()
    sk_clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    sk_clf.fit(X_train, y_train)
    train_time_sklearn = time.time() - start
    print(f"Training time: {train_time_sklearn:.2f} s")
    evaluate(sk_clf, X_val, y_val, "sklearn - Val")
    evaluate(sk_clf, X_test, y_test, "sklearn - Test")

    # 5. Summary
    print("\n=== Summary ===")
    print(f"Custom  training time:  {train_time_custom:.2f} s")
    print(f"sklearn training time: {train_time_sklearn:.2f} s")

if __name__ == "__main__":
    compare_models(
        data_label = "acceptor",
    )
