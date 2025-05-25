
# Polecenie

Drzewo decyzyjne w zadaniu klasyfikacji miejsc rozcięcia w sekwencji DNA. Należy użyć wyrażenia regularnego w testach. Wyrażenia regularne powinny być wczytane z pliku tekstowego. Dopasowanie wyrażenia sprawdzamy na rozważanej pozycji w sekwencji, np. jeśli rozważamy atrybut numer 3, a wyrażenie to "A.T", to w sekwencji "GGAGT" nastąpi dopasowanie, a w sekwencji "AGTGG" już nie. Więcej informacji o specyfice problemu znaleźć można w: [OpisDNA](https://staff.elka.pw.edu.pl/~rbiedrzy/UMA/opisDNA.html). Dane do pobrania- [donory](https://staff.elka.pw.edu.pl/~rbiedrzy/UMA/spliceDTrainKIS.dat), [akceptory](https://staff.elka.pw.edu.pl/~rbiedrzy/UMA/spliceATrainKIS.dat). Przed rozpoczęciem realizacji projektu proszę zapoznać się z zawartością: [LINK](https://staff.elka.pw.edu.pl/~rbiedrzy/UMA/index.html).

### data_loader.py – wczytuje dane z pliku .dat i wycina okno wokół pozycji granicznej.
Import danych

### regex_generator.py – generuje wszystkie możliwe k-gramy z alfabetu {A, C, G, T}.

Generowanie regexów

### model_trainer.py – tworzy cechy (regex, pozycja), dzieli zbiór danych na train/val/test i trenuje drzewo decyzyjne, zapisując model.

Wykorzystanie drzewa decyzyjnego

### /decision_tree - implementacja drzewa decyzyjnego

### main.py - spina to w całość

# How to run:

1. Uruchom main.py – własna implementacja lub sklearn
Plik main.py pozwala trenować i testować klasyfikator DNA na danych typu donor lub acceptor, korzystając z własnego drzewa decyzyjnego lub z sklearn.

Przykład – własne drzewo (domyślnie):

```bash
python main.py --data_type donor --data_path input_data/spliceDTrainKIS.dat
```
Przykład – drzewo z sklearn:

```bash
python main.py --impl sklearn --data_type acceptor --data_path input_data/spliceATrainKIS.dat --max_depth 15
```
Dodatkowe argumenty:

--impl – wybór implementacji: 'custom' lub 'sklearn' (domyślnie 'custom')

--window_size – rozmiar okna DNA (domyślnie 5)

--positions – pozycje do ekstrakcji regexów (domyślnie 7 68)

--min_samples – minimalna liczba próbek do podziału (tylko dla custom)

--random_state – ziarno generatora losowego

--test_size – rozmiar zbioru testowego/walidacyjnego (domyślnie 0.2)

2. Porównanie implementacji (custom vs sklearn)
Aby porównać własne drzewo z implementacją sklearn:

```bash
python -m testing.custom_vs_sklearn
```
Uruchamiaj z głównego katalogu projektu.

Skrypt trenuje oba modele, mierzy czas uczenia i wypisuje metryki walidacyjne i testowe.

3. Strojenie hiperparametrów
Aby automatycznie przeszukać różne wartości max_depth i min_samples (dla własnego drzewa):

```bash
python -m testing.hyperparameter_tuning
```
Siatkę hiperparametrów możesz zmieniać wewnątrz pliku testing/hyperparameter_tuning.py:

```
max_depths = [3, 5, 10, 15, 20]
min_samples_list = [2, 5, 10]
```
Dla każdej kombinacji liczona jest średnia z metryk (precision, recall, F1, accuracy) na podstawie kilku powtórzeń (domyślnie 5).