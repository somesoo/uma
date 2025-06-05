import itertools
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

from src.data_loader import load_dna_with_window
from src.regex_generator import load_regex_patterns
from src.model_trainer import extract_features, extract_features_full
from src.decision_tree.model import DecisionTree


def format_confusion_matrix_with_labels(cm):
    tn, fp, fn, tp = cm.ravel()
    labeled = (
        f"Confusion Matrix:\n"
        f"[[TN={tn}  FP={fp}]\n"
        f" [FN={fn}  TP={tp}]]"
    )
    return labeled


def enhanced_print_classification_report(y_true, y_pred, title, save_dir="reports"):
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=3)

    print(f"\n=== {title} ===")
    print("Confusion Matrix:")
    print(format_confusion_matrix_with_labels(cm))
    print("\n=== Classification report ===")
    print(report)

    # Zapis raportu do pliku
    file_title = title.replace(" ", "_").replace("|", "").lower()
    path = os.path.join(save_dir, f"{file_title}_report.txt")
    with open(path, "w") as f:
        f.write(f"{title}\n\n")
        f.write(format_confusion_matrix_with_labels(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Zapis wykresu confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(title)
    plt.savefig(os.path.join(save_dir, f"{file_title}_confusion_matrix.png"))
    plt.close()


def compare_window_vs_full():
    configs = list(itertools.product(
        ["donor", "acceptor"],       # typ danych
        ["window", "full"],          # metoda ekstrakcji regexów
        ["custom", "sklearn"]        # implementacja drzewa
    ))

    regex_paths = {
        "donor": "input_data/regex_donor.txt",
        "acceptor": "input_data/regex_acceptor.txt"
    }   
    data_paths = {
        "donor": "input_data/spliceDTrainKIS.dat",
        "acceptor": "input_data/spliceATrainKIS.dat"
    }

    random_state = 12
    test_size = 0.2
    max_depth = 25
    min_samples = 2

    results = []

    for data_type, feature_type, impl in configs:
        regexes = load_regex_patterns(regex_paths[data_type])
        examples = load_dna_with_window(data_paths[data_type], data_type)

        if feature_type == "full":
            X, y = extract_features_full(examples, regexes, output_file="features_full.csv")
        else:
            X, y = extract_features(examples, regexes, output_file="features_window.csv")

        X_trval, X_test, y_trval, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_trval, y_trval, test_size=test_size, random_state=random_state, stratify=y_trval)

        if impl == "custom":
            model = DecisionTree(max_depth=max_depth, min_samples=min_samples, n_feats=X.shape[1], random_state=random_state)
        else:
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        title = f"{data_type} | {impl} | {feature_type}"
        enhanced_print_classification_report(y_test, y_pred, title)

        results.append({
            "data_type": data_type,
            "implementation": impl,
            "feature_type": feature_type,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
        })

    df = pd.DataFrame(results)
    df_pivot = df.pivot_table(
        index=["data_type", "implementation"],
        columns="feature_type",
        values=["accuracy", "precision", "recall", "f1_score"]
    )

    df_pivot.columns = [f"{metric}_{ftype}" for metric, ftype in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    print("\n=== Summary: Full vs Window ===")
    print(df_pivot.to_string(index=False))


if __name__ == "__main__":
    compare_window_vs_full()