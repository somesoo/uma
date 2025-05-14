import numpy as np
from itertools import product
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.data_loader import load_dna_with_window
from src.regex_generator import load_regex_patterns
from src.model_trainer import extract_features
from src.decision_tree.model import DecisionTree

def tune_hyperparameters(
    donor_path: str,
    acceptor_path: str,
    regex_path: str,
    window_size: int,
    positions: list[int],
    max_depths: list[int],
    min_samples_list: list[int],
    test_size: float = 0.2,
    random_state: int = 42
):
    # 1) Wczytaj i przygotuj cechy
    donor   = load_dna_with_window(donor_path,   "donor",    window_size)
    acceptor= load_dna_with_window(acceptor_path,"acceptor", window_size)
    examples= donor + acceptor
    regexes = load_regex_patterns(regex_path)
    X, y    = extract_features(examples, regexes, positions)

    # 2) Split: train+val / test, a następnie train / val
    X_trval, X_test,  y_trval, y_test  = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trval, y_trval, test_size=test_size, random_state=random_state, stratify=y_trval
    )

    # 3) Grid search po max_depth i min_samples
    results = []
    for md, ms in product(max_depths, min_samples_list):
        clf = DecisionTree(
            max_depth   = md,
            min_samples = ms,
            n_feats     = int(np.sqrt(X.shape[1])),
            random_state= random_state
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        acc    = accuracy_score(y_val, y_pred)
        results.append((md, ms, acc))

    # 4) Sortuj i wypisz
    results.sort(key=lambda x: x[2], reverse=True)
    best_md, best_ms, best_acc = results[0]
    print(f"\nBEST → max_depth={best_md}, min_samples={best_ms}, val_acc={best_acc:.3f}\n")
    print("ALL results:")
    for md, ms, acc in results:
        print(f"  depth={md:2d}, min_samp={ms:2d} → val_acc={acc:.3f}")

if __name__ == "__main__":
    tune_hyperparameters(
        donor_path       = "input_data/spliceDTrainKIS.dat",
        acceptor_path    = "input_data/spliceATrainKIS.dat",
        regex_path       = "input_data/regex_patterns.txt",
        window_size      = 5,
        positions        = [7, 68],
        max_depths       = [3, 5, 10, 15, 20, 30],
        min_samples_list = [2, 5, 10, 20, 30],
        test_size        = 0.2,
        random_state     = 42
    )