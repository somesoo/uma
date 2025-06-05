import itertools
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.data_loader import load_dna_with_window
from src.regex_generator import load_regex_patterns
from src.model_trainer import extract_features, extract_features_full, extract_one_hot_features
from src.decision_tree.model import DecisionTree


def compare_onehot_vs_regex():
    configs = list(itertools.product(
        ["donor", "acceptor"],       # data_type
        ["regex", "onehot"],         # feature_type
        ["custom", "sklearn"]        # implementation
    ))

    regex_paths = {
        "donor": "input_data/regex_donor.txt",
        "acceptor": "input_data/regex_acceptor.txt"
    }  
    data_paths = {
        "donor": "input_data/spliceDTrainKIS.dat",
        "acceptor": "input_data/spliceATrainKIS.dat"
    }

    random_state = 42
    test_size = 0.2
    max_depth = 30
    min_samples = 2

    results = []

    for data_type, feature_type, impl in configs:
        regexes = load_regex_patterns(regex_paths[data_type])
        regex_len = len(regexes[1])
        examples = load_dna_with_window(data_paths[data_type], data_type)

        if feature_type == "regex":
            X, y = extract_features(examples, regexes)
        else:
            X, y, _ = extract_one_hot_features(examples)

        X_trval, X_test, y_trval, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_trval, y_trval, test_size=test_size, random_state=random_state, stratify=y_trval)

        if impl == "custom":
            model = DecisionTree(max_depth=max_depth, min_samples=min_samples, n_feats=X.shape[1], random_state=random_state)
        else:
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            "data_type": data_type,
            "implementation": impl,
            "feature_type": feature_type,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
        })

    df = pd.DataFrame(results)
    df_pivot = df.pivot_table(
        index=["data_type", "implementation"],
        columns="feature_type",
        values=["accuracy", "precision", "recall", "f1_score"]
    )

    df_pivot.columns = [f"{metric}_{ftype}" for metric, ftype in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    print("\n=== One-hot vs Regex Summary ===")
    print(df_pivot.to_string(index=False))


if __name__ == "__main__":
    compare_onehot_vs_regex()