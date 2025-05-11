import re
from typing import List, Tuple, Optional

# Typ danych: (etykieta, sekwencja, typ zbioru)
Example = Tuple[int, str, str]

def load_dna_dataset(filepath: str, dataset_type: str) -> List[Example]:
    """
    Wczytuje plik z danymi DNA (donory lub akceptory).
    
    :param filepath: ścieżka do pliku
    :param dataset_type: "donor" lub "acceptor"
    :return: lista przykładów (etykieta, sekwencja, typ)
    """
    examples = []
    with open(filepath, "r") as file:
        lines = file.read().splitlines()
        lines = lines[1:]  # pomijamy pierwszą linię

        for i in range(0, len(lines), 2):
            label = int(lines[i].strip())
            sequence = lines[i + 1].strip().upper()
            examples.append((label, sequence, dataset_type))
    return examples

def load_all_dna_datasets(donor_path: Optional[str] = None,
                          acceptor_path: Optional[str] = None) -> List[Example]:
    """
    Ładuje dane z plików donorów i/lub akceptorów, jeśli zostały podane.

    :param donor_path: ścieżka do pliku donorów
    :param acceptor_path: ścieżka do pliku akceptorów
    :return: lista wszystkich przykładów z etykietą typu
    """
    all_examples = []

    if donor_path:
        all_examples += load_dna_dataset(donor_path, "donor")
    if acceptor_path:
        all_examples += load_dna_dataset(acceptor_path, "acceptor")

    return all_examples

def load_regex_patterns(filepath: str) -> List[str]:
    """
    Wczytuje wyrażenia regularne z pliku tekstowego.
    Każda linia to jedno wyrażenie.

    :param filepath: ścieżka do pliku z regexami
    :return: lista wzorców jako stringi
    """
    with open(filepath, "r") as file:
        return [line.strip() for line in file if line.strip()]

def regex_matches_at_position(sequence: str, pattern: str, position: int) -> bool:
    """
    Sprawdza, czy regex pasuje do fragmentu sekwencji DNA na podanej pozycji.

    :param sequence: sekwencja DNA
    :param pattern: wyrażenie regularne
    :param position: pozycja początkowa
    :return: True jeśli pasuje, False w przeciwnym razie
    """
    end = position + len(pattern)
    if end > len(sequence):
        return False
    subseq = sequence[position:end]
    return re.fullmatch(pattern, subseq) is not None

# Testujemy funkcje wczytujące
donor_path = "spliceDTrainKIS.dat"
acceptor_path = "spliceATrainKIS.dat"
regex_file = "regex_patterns.txt"

try:
    examples = load_all_dna_datasets(donor_path, acceptor_path)
    print(f"Wczytano {len(examples)} przykładów")

    regex_list = load_regex_patterns(regex_file)
    print(f"Wczytano {len(regex_list)} regexów")

    # Przykładowe sprawdzenie dopasowania
    test_seq = examples[0][1]
    print("Sekwencja:", test_seq)
    for r in regex_list:
        print(f"Regex '{r}' na pozycji 3:", regex_matches_at_position(test_seq, r, 3))

except FileNotFoundError as e:
    print("Brak pliku:", e)
except Exception as e:
    print("Błąd:", e)

