## data_loader.py – wczytuje dane z pliku .dat i wycina okno wokół pozycji granicznej.


##  regex_generator.py – generuje wszystkie możliwe k-gramy z alfabetu {A, C, G, T}.

To jest plik w którym będzie trzeba stworzyć odpowiednie regexy, aby zoptymalizować uczenie


##  model_trainer.py – tworzy cechy (regex, pozycja), dzieli zbiór danych na train/val/test i trenuje drzewo decyzyjne, zapisując model.

To jest plik gdzie trzeba zrobić fajną implementacje tego drzewa, bo raczej sklearn nie zadziała aby dostać dobrą ocenę


## model_predictor.py – umożliwia załadowanie modelu z pliku i (opcjonalnie) predykcję.