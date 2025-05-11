from typing import List, Tuple

Example = Tuple[int, str, str, str]  # (label, window_seq, dataset_type, full_seq)

def load_dna_with_window(filepath: str, dataset_type: str, window_size: int = 5) -> List[Example]:
    """
    Wczytuje plik z sekwencjami DNA. Wyciąga okno wokół granicy intron-ekson.

    :param filepath: Ścieżka do pliku z danymi
    :param dataset_type: Typ zbioru ("donor" lub "acceptor")
    :param window_size: liczba znaków po lewej i prawej stronie pozycji granicznej
    :return: lista krotek (etykieta, okno sekwencji, typ, pełna sekwencja)
    """
    examples = []
    with open(filepath, "r") as f:
        lines = f.read().splitlines()
        boundary_pos = int(lines[0].strip())  # pozycja graniczna
        lines = lines[1:]  # reszta to dane

        for i in range(0, len(lines), 2):
            label = int(lines[i].strip())
            full_seq = lines[i + 1].strip().upper()
            start = max(0, boundary_pos - window_size)
            end = boundary_pos + window_size
            window_seq = full_seq[start:end]
            examples.append((label, window_seq, dataset_type, full_seq))
    return examples


# Przykład użycia:
donor_file = "spliceDTrainKIS.dat"
acceptor_file = "spliceATrainKIS.dat"

try:
    donor_examples = load_dna_with_window(donor_file, "donor", window_size=5)
    acceptor_examples = load_dna_with_window(acceptor_file, "acceptor", window_size=5)

    print(f"Wczytano {len(donor_examples)} przykładów donorów")
    print(f"Wczytano {len(acceptor_examples)} przykładów akceptorów")
    print("Przykład donora:", donor_examples[0])
    print("Przykład akceptora:", acceptor_examples[0])

except FileNotFoundError as e:
    print("Brak pliku:", e)
except Exception as e:
    print("Błąd:", e)

