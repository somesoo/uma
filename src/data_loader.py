from typing import List, Tuple

Example = Tuple[int, str, str, str]

def load_dna_with_window(filepath: str, dataset_type: str, regex_len: str) -> List[Example]:
    examples = []
    with open(filepath, "r") as f:
        lines = f.read().splitlines()
        boundary_pos = int(lines[0].strip()) - 1
        lines = lines[1:]

        for i in range(0, len(lines), 2):
            label = int(lines[i].strip())
            full_seq = lines[i + 1].strip().upper()
            start = boundary_pos
            end = boundary_pos + regex_len
            if end <= len(full_seq):  # zabezpieczenie
                window_seq = full_seq[start:end]
                examples.append((label, window_seq, dataset_type, full_seq))
            else:
                print(f"Ostrzeżenie: sekwencja z etykietą {label} za krótka (len={len(full_seq)}), pominięto.")


        # for i in range(0, len(lines), 2):
        #     label = int(lines[i].strip())
        #     full_seq = lines[i + 1].strip().upper()
        #     start = boundary_pos
        #     end = min(boundary_pos + 10, len(full_seq))  # zabezpieczenie przed wyjściem poza sekwencję
        #     window_seq = full_seq[start:end]

        #     if len(window_seq) < 3:  # lub inny minimalny próg, np. dla 3-literowych regexów
        #         print(f"Ostrzeżenie: sekwencja z etykietą {label} za krótka ({len(window_seq)} znaków), pominięto.")
        #         continue

        #     examples.append((label, window_seq, dataset_type, full_seq))
    return examples