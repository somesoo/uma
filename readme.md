# Polecenie

Drzewo decyzyjne w zadaniu klasyfikacji miejsc rozcięcia w sekwencji DNA. Należy użyć wyrażenia regularnego w testach. Wyrażenia regularne powinny być wczytane z pliku tekstowego. Dopasowanie wyrażenia sprawdzamy na rozważanej pozycji w sekwencji, np. jeśli rozważamy atrybut numer 3, a wyrażenie to "A.T", to w sekwencji "GGAGT" nastąpi dopasowanie, a w sekwencji "AGTGG" już nie. Więcej informacji o specyfice problemu znaleźć można w: https://staff.elka.pw.edu.pl/~rbiedrzy/UMA/opisDNA.html. Dane do pobrania- donory: https://staff.elka.pw.edu.pl/~rbiedrzy/UMA/spliceDTrainKIS.dat, akceptory: https://staff.elka.pw.edu.pl/~rbiedrzy/UMA/spliceATrainKIS.dat. Przed rozpoczęciem realizacji projektu proszę zapoznać się z zawartością: https://staff.elka.pw.edu.pl/~rbiedrzy/UMA/index.html.

### data_loader.py – wczytuje dane z pliku .dat i wycina okno wokół pozycji granicznej.
Import danych

### regex_generator.py – generuje wszystkie możliwe k-gramy z alfabetu {A, C, G, T}.

To jest plik w którym będzie trzeba stworzyć odpowiednie regexy, aby zoptymalizować uczenie, na razie jest basic k-gram


### model_trainer.py – tworzy cechy (regex, pozycja), dzieli zbiór danych na train/val/test i trenuje drzewo decyzyjne, zapisując model.

To jest plik gdzie trzeba zrobić fajną implementacje tego drzewa, bo raczej sklearn nie zadziała aby dostać dobrą ocenę

### model_predictor.py – umożliwia załadowanie modelu z pliku i (opcjonalnie) predykcję.

### main.py - spina to w całość