from src.data_loader import load_dna_with_window
from src.regex_generator import save_kgram_regexes_to_file, load_regex_patterns
from src.model_trainer import extract_features, train_and_save_model

if __name__ == "__main__":
    # Wczytaj dane
    donor_examples = load_dna_with_window("input_data/spliceDTrainKIS.dat", "donor", window_size=5)
    acceptor_examples = load_dna_with_window("input_data/spliceATrainKIS.dat", "acceptor", window_size=5)
    all_examples = donor_examples + acceptor_examples

    # Generuj regexy długości 3
    save_kgram_regexes_to_file(3, "regex_patterns.txt")
    regex_list = load_regex_patterns("regex_patterns.txt")

    # Pozycje testowe
    positions = [7, 68]  # zgodnie z poleceniem

    # Ekstrakcja cech
    X, y = extract_features(acceptor_examples, regex_list, positions)

    # Trening i zapis modelu
    train_and_save_model(X, y, model_path="output/acceptor_decision_tree_model.joblib")
