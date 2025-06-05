import os
import re
import joblib
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from src.decision_tree.model import DecisionTree
from typing import List, Tuple
from sklearn.metrics import classification_report

Example = Tuple[int, str, str, str]

def extract_features_full(
    examples: List[Example],
    regex_list: List[str],
    output_file: str = "features.csv"
) -> Tuple[np.ndarray, np.ndarray]:
    
    X, y = [], []
    header = [f"rx_{i}_{rx}" for i, rx in enumerate(regex_list)]
    
    with open(output_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header + ["label"])

        for label, _, _, full_seq in examples:
            feats = []
            for rx in regex_list:
                regex_len = len(rx)
                found = False
                for i in range(len(full_seq) - regex_len + 1):
                    fragment = full_seq[i:i+regex_len]
                    if re.fullmatch(rx, fragment):
                        found = True
                        break
                feats.append(int(found))
            writer.writerow(feats + [label])
            X.append(feats)
            y.append(label)

    return np.array(X, dtype=int), np.array(y, dtype=int)

def extract_features(
    examples: List[Example],
    regex_list: List[str],
    output_file: str = "features.csv"
) -> Tuple[np.ndarray, np.ndarray]:
    
    X, y = [], []
    header = [f"rx_{i}_{rx}" for i, rx in enumerate(regex_list)]
    
    with open(output_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header + ["label"]) 
        
        for label, window_seq, *_ in examples:
            feats = [
                int(re.match(rx, window_seq) is not None)
                for rx in regex_list
            ]
            writer.writerow(feats + [label])
            X.append(feats)
            y.append(label)
    
    return np.array(X, dtype=int), np.array(y, dtype=int)

def extract_one_hot_features(examples):
    nucleotides = ["A", "C", "G", "T"]
    X = []
    y = []

    seq_len = len(examples[0][1])
    feature_names = [f"pos{i}_{nuc}" for i in range(seq_len) for nuc in nucleotides]

    for label, seq, _, _ in examples:
        row = []
        for char in seq:
            for nuc in nucleotides:
                row.append(1 if char == nuc else 0)
        X.append(row)
        y.append(label)
        #print(row, label)
    return np.array(X), np.array(y), feature_names


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.3f}")
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        print(f"AUC:       {auc(fpr, tpr):.3f}")
        plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(loc="lower right")
#        plt.show()
    except AttributeError:
        print("Model does not support probability predictions.")
    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred, digits=3))
    
def train_and_save_model(
    X, y,
    model_path: str,
    max_depth: int = 10,
    min_samples: int = 2,
    random_state: int = 30
):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    X = np.asarray(X)
    y = np.asarray(y)

    X_trval, X_test, y_trval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trval, y_trval, test_size=0.2, random_state=random_state, stratify=y_trval
    )

    clf = DecisionTree(
        max_depth=max_depth,
        min_samples=min_samples,
        n_feats=int(np.sqrt(X.shape[1])),
        random_state=random_state
    )
    clf.fit(X_train, y_train)

    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}\n")

    print("=== Validation set ===")
    evaluate(clf, X_val, y_val)
    print("\n=== Test set ===")
    evaluate(clf, X_test, y_test)
