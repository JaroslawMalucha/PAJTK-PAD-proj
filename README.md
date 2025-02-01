# PAD - Projekt - Fraud Detection

## Cel i zakres projektu

Celem projektu była próba stworzenia modelu predykcji czy dana transakcja jest fraudem czy nie.
Wykorzystano do tego celu dataset dostępny na platformie Kaggle, który zawiera dane transakcji i informację czy ta transakcja jest fraudem.
Przeprowadzono analizę danych w celu poszukiwania występujących wzorców.
W ramach przygotowania danych do modelu dokonano licznych transformacji, czyszczenie i feature engineering, a w tym, m.in. sprawdzenie wartości brakujących i odstających, usunięcie zbędnych cech,
wyliczenie nowych zmiennych, transformacje one-hot-encoding, przeskalowanie, resampling i rebalancing.
Jako klasyfikator użyto Gradient Boost Classifier.

Prace zostały zapisane w postaci skryptów oraz towarzyszącym im notebooków Jupyter `.ipynb` kopiujących funkcjonalność zwykłych skryptów `.py`.

Repozytorium projektu znajduje się na platformie GitHub.

## Wyniki

Zbudowany model osiągnął wysoki poziom wykrywalności fraudów, 
ale również i bardzo wysoki poziom wyników false-positive,
czyli takich gdzie transakcje nie będąca fraudem została zakwalifikowana jako fraud.
Taki model być może mógłby być wykorzystania do pomocnicznego zaflagowania transakcji,
ale nie może być wykorzystany do automatycznego ich blokowania, 
gdyż spowodowałoby to ogromne niezadowolenie klientów.

W poprawy wyników modelu można, m.in. poeksperymentować z dalszą pracą w zakresie inżynierii cech oraz spróbować zastosować inne typy modeli klasyfikację takie jak np. XGBoost.

## Struktura projektu

`data` - folder dla danych wejściowych

`data/credit-data.csv` - plik wejściowy pobrany z platformy Kaggle (wymaga pobrania)

`src` - folder z kodem źródłowym

[`src/source_data.py`](src/source_data.py) - skrypt do pobrania danych z Kaggle (wymaga konfiguracji tokena w `kaggle.json`)

[`src/eda.py`](src/eda.py) - analiza i eksploracja danych

[`src/eda.ipynb`](src/eda.ipynb) - analiza i eksploracja danych

[`src/data_and_model.py`](src/data_and_model.py) - przygotowanie danych, trenowanie modelu i ewaluacja

[`src/data_and_model.ipynb`](src/data_and_model.py) - przygotowanie danych, trenowanie modelu i ewaluacja

[`.gitignore`](.gitignore) - reguły ignorowania dla git'a

[`packages_deploy_app.ps1`](packages_deploy_app.ps1) - instalacja paczek python i środowisko wirtualne

[`requirements-app.txt`](requirements-app.txt) - zestawienie użytych paczek python

## Uruchomienie projektu

- użyj python 3.11
- należy stworzyć plik z tokenem z Kaggle w korzeniu repozytorium - `kaggle.json`
- zainstaluj paczki zgodnie z informacjami w `packages_deploy_app.ps1`
- użyj `src/source_data.py` aby pobrać dane
- użyj `src/eda.py` aby zobaczyć analizę
- użyj `src/data_and_model.py` aby uruchomić transformacje danych i wytrenować model
- lub użyj odpowiedników w formacie `.ipynb`

## Dane wejściowe

[link na Kaggle](https://www.kaggle.com/datasets/kelvinkelue/credit-card-fraud-prediction)

## GitHub

[link do repozytorium na GitHub](https://github.com/JaroslawMalucha/PAJTK-PAD-proj)
