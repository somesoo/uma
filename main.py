from data_loader import load_dna_with_window
from regex_generator import generate_kgram_regexes
from model_trainer import extract_features, train_and_save_model

if __name__ == "__main__":
    # Wczytaj dane
    donor_examples = load_dna_with_window("spliceDTrainKIS.dat", "donor", window_size=5)
    acceptor_examples = load_dna_with_window("spliceATrainKIS.dat", "acceptor", window_size=5)
    all_examples = donor_examples + acceptor_examples

    # Generuj regexy długości 3
    regex_list = generate_kgram_regexes(3)

    # Pozycje testowe
    positions = [7, 68]  # zgodnie z poleceniem

    # Ekstrakcja cech
    X, y = extract_features(all_examples, regex_list, positions)

    # Trening i zapis modelu
    train_and_save_model(X, y, model_path="decision_tree_model.joblib")