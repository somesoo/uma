### Autor: Radosław Kasprzak & Jonatan Kasperczak

import numpy as np
from itertools import product
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from src.data_loader import load_dna_with_window
from src.regex_generator import load_regex_patterns
from src.model_trainer import extract_features, extract_features_full
from src.decision_tree.model import DecisionTree

def tune_hyperparameters(
    data_path: str,
    data_label: str,
    regex_path: str,
    max_depths: list[int],
    min_samples_list: list[int],
    test_size: float = 0.2,
    random_state: int = 12,
    n_repeats: int = 5
):
    if data_label.lower() == "acceptor":
        positions = [68]
    elif data_label.lower() == "donor":
        positions = [7]
    else:
        raise ValueError(f"Unsupported data_label: {data_label}")

    print(f"\nTuning hyperparameters for '{data_label}' (positions = {positions})")

    regexes  = load_regex_patterns(regex_path)
    examples = load_dna_with_window(data_path, data_label)
    X, y     = extract_features(examples, regexes)

    X_trval, X_test,  y_trval, y_test  = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trval, y_trval, test_size=test_size, random_state=random_state, stratify=y_trval
    )

    results = []
    for md, ms in product(max_depths, min_samples_list):

        metrics = []

        for repeat in range(n_repeats):

            X_trval, X_test, y_trval, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state + repeat, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_trval, y_trval, test_size=test_size, random_state=random_state + repeat, stratify=y_trval
            )

            clf = DecisionTree(
                max_depth=md,
                min_samples=ms,
                n_feats=int(np.sqrt(X.shape[1])),
                random_state=random_state + repeat
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)

            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            metrics.append((acc, prec, rec, f1, tp, fp, tn, fn))

        # Average
        metrics = np.array(metrics)
        mean_metrics = metrics.mean(axis=0)
        results.append((md, ms, *mean_metrics))

    results.sort(key=lambda x: (x[4], x[2]), reverse=True)

    print(f"\nBEST for {data_label} → max_depth={results[0][0]}, min_samples={results[0][1]}, Recall={results[0][4]:.2f}, F1={results[0][5]:.2f}\n")
    print("ALL results:")
    for res in results:
        md, ms, acc, prec, rec, f1, tp, fp, tn, fn = res
        print(f"depth={md:2d}, min_samp={ms:2d} → acc={acc:.2f}, prec={prec:.2f}, rec={rec:.2f}, f1={f1:.2f}")

if __name__ == "__main__":
    
    type = ["donor", "acceptor"]
    file = ["input_data/spliceDTrainKIS.dat", "input_data/spliceATrainKIS.dat"]
    regex_paths = ["input_data/regex_donor.txt","input_data/regex_acceptor.txt"]
    for i, j, k in zip(type, file, regex_paths):
        tune_hyperparameters(
            data_path        = j,
            data_label       = i,
            regex_path       = k,
            max_depths       = [3, 5, 10, 15, 20, 30],
            min_samples_list = [2, 5, 10, 20, 30],
            test_size        = 0.2,
            random_state     = 12,
            n_repeats        = 10
        )
