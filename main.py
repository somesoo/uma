from src.data_loader import load_dna_with_window
from src.regex_generator import save_kgram_regexes_to_file, load_regex_patterns
from src.model_trainer import extract_features, train_and_save_model
from collections import Counter


import numpy as np
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from src.data_loader import load_dna_with_window
from src.regex_generator import load_regex_patterns
from src.model_trainer import extract_features, evaluate
from src.decision_tree.model import DecisionTree

# if __name__ == "__main__":
#     # Wczytaj dane
#     donor_examples = load_dna_with_window("input_data/spliceDTrainKIS.dat", "donor", window_size=5)
#     acceptor_examples = load_dna_with_window("input_data/spliceATrainKIS.dat", "acceptor", window_size=5)
#     all_examples = donor_examples + acceptor_examples

#     # Generuj regexy długości 3
#     # save_kgram_regexes_to_file(3, "input_data/regex_patterns.txt")
#     regex_list = load_regex_patterns("input_data/regex_patterns.txt")

#     # Pozycje testowe
#     positions = [7, 68]  # zgodnie z poleceniem

#     # Ekstrakcja cech
#     X, y = extract_features(acceptor_examples, regex_list, positions)
#     print("Class balance:", Counter(y))
#     # Trening i zapis modelu
#     train_and_save_model(X, y, model_path="output/acceptor_decision_tree_model.joblib")


if __name__ == "__main__":
    # 1) Wczytaj dane
    donor_examples    = load_dna_with_window("input_data/spliceDTrainKIS.dat", "donor",    window_size=5)
    acceptor_examples = load_dna_with_window("input_data/spliceATrainKIS.dat", "acceptor", window_size=5)
    all_examples      = donor_examples + acceptor_examples

    # 2) Regexy i pozycje
    regex_list = load_regex_patterns("input_data/regex_patterns.txt")
    positions  = [7, 68]

    # 3) Ekstrakcja cech
    X, y = extract_features(all_examples, regex_list, positions)
    print("Class balance:", Counter(y))

    # 4) Split: train+val / test, a potem train / val
    X_trval, X_test,  y_trval, y_test  = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trval, y_trval, test_size=0.2, random_state=42, stratify=y_trval
    )

    # 5a) Własne drzewo
    custom_clf = DecisionTree(
        max_depth=10,
        min_samples=2,
        n_feats=int(np.sqrt(X.shape[1])),
        random_state=42
    )
    custom_clf.fit(X_train, y_train)
    print("\n=== Custom DecisionTree ===")
    print("Validation:")
    evaluate(custom_clf, X_val, y_val)
    print("\nTest:")
    evaluate(custom_clf, X_test, y_test)

    # 5b) sklearn DecisionTreeClassifier
    sk_clf = DecisionTreeClassifier(max_depth=10, random_state=42)
    sk_clf.fit(X_train, y_train)
    print("\n=== sklearn DecisionTreeClassifier ===")
    print("Validation:")
    evaluate(sk_clf, X_val, y_val)
    print("\nTest:")
    evaluate(sk_clf, X_test, y_test)