# Polecenie

Drzewo decyzyjne w zadaniu klasyfikacji miejsc rozcięcia w sekwencji DNA. Należy użyć wyrażenia regularnego w testach. Wyrażenia regularne powinny być wczytane z pliku tekstowego. Dopasowanie wyrażenia sprawdzamy na rozważanej pozycji w sekwencji, np. jeśli rozważamy atrybut numer 3, a wyrażenie to "A.T", to w sekwencji "GGAGT" nastąpi dopasowanie, a w sekwencji "AGTGG" już nie. Więcej informacji o specyfice problemu znaleźć można w: [OpisDNA](https://staff.elka.pw.edu.pl/~rbiedrzy/UMA/opisDNA.html). Dane do pobrania- [donory](https://staff.elka.pw.edu.pl/~rbiedrzy/UMA/spliceDTrainKIS.dat), [akceptory](https://staff.elka.pw.edu.pl/~rbiedrzy/UMA/spliceATrainKIS.dat). Przed rozpoczęciem realizacji projektu proszę zapoznać się z zawartością: [LINK](https://staff.elka.pw.edu.pl/~rbiedrzy/UMA/index.html).

---

## Struktura projektu

### `data_loader.py`  
Wczytuje dane z pliku `.dat`, wycina okno wokół pozycji granicznej.  
**Import danych.**

### `model_trainer.py`  
Tworzy cechy `(regex, pozycja)`, dzieli dane na `train/val/test`, trenuje drzewo decyzyjne, zapisuje model.  
**Wykorzystanie drzewa decyzyjnego.**

### `/decision_tree/`  
Implementacja własnego drzewa decyzyjnego.

### `main.py`  
Spina cały pipeline w całość.

---

# Jak uruchomić

### Uruchomienie `main.py` — własna implementacja lub sklearn

Plik `main.py` umożliwia trenowanie i testowanie klasyfikatora DNA na danych `donor` lub `acceptor`, korzystając z własnego drzewa decyzyjnego lub z `sklearn`.

**Przykład — własne drzewo (domyślnie):**

```bash
python main.py --data_type donor
```

**Przykład — drzewo z sklearn:**

```bash
python main.py --impl sklearn --data_type acceptor --max_depth 15
```

### Dodatkowe argumenty `main.py`

| Argument | Wartości | Domyślna wartość | Opis |
|----------|----------|------------------|------|
| `--data_type` | `donor`, `acceptor` | **(wymagany)** | Typ danych wejściowych |
| `--data_path` | ścieżka do pliku | *(opcjonalnie)* | Ścieżka do pliku z danymi DNA |
| `--regex_path` | ścieżka do pliku | `input_data/regex_patterns.txt` | Ścieżka do pliku z wyrażeniami regularnymi |
| `--window_size` | liczba całkowita | `3` | Rozmiar okna do wycinania fragmentu sekwencji DNA |
| `--max_depth` | liczba całkowita | `30` | Maksymalna głębokość drzewa decyzyjnego |
| `--min_samples` | liczba całkowita | `2` | Minimalna liczba próbek do podziału (tylko dla `custom`) |
| `--test_size` | liczba zmiennoprzecinkowa (0..1) | `0.2` | Procent danych przeznaczony na testy |
| `--random_state` | liczba całkowita | `12` | Ziarno generatora losowego |
| `--impl` | `custom`, `sklearn` | `custom` | Implementacja drzewa decyzyjnego do użycia |
| `--feature_type` | `regex`, `onehot` | `regex` | Typ cech: wyrażenia regularne lub kodowanie one-hot |
| `--regex_search` | `full`, `window` | `window` | Tryb ekstrakcji regexów: z całej sekwencji lub tylko z okna |
| `--oversample` | *(flaga)* | `False` | Jeżeli ustawione, stosuje oversampling klasy mniejszościowej przed treningiem |

---

### Porównanie implementacji (custom vs sklearn)

Aby porównać własne drzewo z implementacją `sklearn`:

```bash
python -m testing.custom_vs_sklearn
```

Uruchamiać z głównego katalogu projektu.

**Opis:**  
Skrypt trenuje oba modele, mierzy czas uczenia i wypisuje metryki walidacyjne i testowe.

---

### Strojenie hiperparametrów

Aby automatycznie przeszukać różne wartości `max_depth` i `min_samples` (dla własnego drzewa):

```bash
python -m testing.hyperparameter_tuning
```

**Konfiguracja siatki hiperparametrów:** (zmieniamy w pliku `testing/hyperparameter_tuning.py`):

```python
max_depths = [3, 5, 10, 15, 20]
min_samples_list = [2, 5, 10]
```

**Opis:**  
Dla każdej kombinacji liczona jest średnia metryk (`precision`, `recall`, `F1`, `accuracy`) na podstawie kilku powtórzeń (domyślnie `10`).

---

### Iteracyjne ulepszanie zbioru regexów (pełny cykl)

Przykład pętli bash do uruchomienia wielu iteracji:

```bash
for i in {1..5000}; do
    LOGFILE="logs/iteration_${i}.log"
    echo "--- Running main.py (iteration $i) ---" | tee "$LOGFILE"
    python3 main.py --data_type acceptor | tee -a "$LOGFILE"
    
    echo "--- Evaluating regex performance ---" | tee -a "$LOGFILE"
    python3 -m testing.regex_common --save_scores | tee -a "$LOGFILE"
    
    echo "--- Selecting top regexes ---" | tee -a "$LOGFILE"
    python3 select_top_regexes.py | tee -a "$LOGFILE"
done
```

Opis:  
Automatyczna pętla:

- uruchamia model na zbiorze `acceptor`,
- ocenia skuteczność regexów (`regex_common`),
- selekcjonuje najlepsze regexy (`select_top_regexes.py`).

---