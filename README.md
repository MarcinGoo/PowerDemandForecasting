# CenyEnergi ⚡📊

Projekt poświęcony pobieraniu, analizie eksploracyjnej (EDA) oraz modelowaniu i prognozowaniu danych z **Krajowego Systemu Elektroenergetycznego (KSE)** dostarczanych przez **Polskie Sieci Elektroenergetyczne (PSE)**.

---

## 📌 Cel Projektu

Głównym celem projektu jest analiza dynamiki zapotrzebowania na energię elektryczną w Polsce, produkcja ze źródeł odnawialnych (fotowoltaika i wiatr) oraz badanie ich wpływu na zapotrzebowanie pokrywane przez konwencjonalne jednostki wytwórcze oraz kształtowanie się cen energii.

---

## 🗂️ Struktura Projektu

```text
CenyEnergi/
├── data/
│   ├── raw/
│   │   └── dane_energetyczne_pse.csv   # Surowe dane pobrane z API PSE
│   └── processed/                      # Przetworzone zbiory danych i cechy
├── notebooks/
│   └── 01_pobieranie_i_eda.ipynb       # Analiza eksploracyjna danych (EDA) i wykresy
├── src/
│   ├── __init__.py
│   └── data_fetcher.py                 # Skrypt i moduł do pobierania danych z API PSE
├── .gitignore                          # Ignorowane pliki i katalogi tymczasowe
├── README.md                           # Dokumentacja projektu
└── requirements.txt                    # Zależności bibliotek Python
```

---

## 📈 Opis Zmiennych

| Zmienna | Opis | Jednostka |
| :--- | :--- | :--- |
| `dtime` | Znacznik czasu pomiaru | `YYYY-MM-DD HH:MM:SS` |
| `demand` | Rzeczywiste krajowe zapotrzebowanie na moc | **MW** |
| `pv` | Generacja z farm fotowoltaicznych | **MW** |
| `wi` | Generacja z farm wiatrowych | **MW** |
| `jg` | Generacja jednostek grafikowych / konwencjonalnych | **MW** |
| `oze_total` | Sumaryczna generacja z OZE (`pv + wi`) | **MW** |
| `net_demand` | Zapotrzebowanie netto na moc konwencjonalną (`demand - oze_total`) | **MW** |

---

## 🌐 Źródło Danych

Dane pobierane są bezpośrednio z publicznego API PSE:
* **Punkt końcowy:** `https://api.raporty.pse.pl/api/his-wlk-cal`
* Pobieranie realizowane jest w interwałach czasowych z obsługą paginacji (`nextLink`).

---

## 🚀 Szybki Start

### 1. Klonowanie repozytorium
```bash
git clone https://github.com/MarcinGoo/CenyEnergi.git
cd CenyEnergi
```

### 2. Utworzenie środowiska wirtualnego
```bash
python -m venv venv
# Linux / WSL / macOS:
source venv/bin/activate
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
```

### 3. Instalacja zależności
```bash
pip install -r requirements.txt
```

### 4. Pobieranie danych z API
Możesz uruchomić skrypt CLI do pobrania najnowszych danych:
```bash
python src/data_fetcher.py --days 30 --output data/raw/dane_energetyczne_pse.csv
```

### 5. Uruchomienie Jupyter Notebook
```bash
jupyter notebook notebooks/01_pobieranie_i_eda.ipynb
```

---

## 🗺️ Roadmapa / Kolejne Kroki
- [x] Automatyzacja pobierania danych z API PSE
- [x] Wstępna analiza eksploracyjna danych (EDA)
- [ ] Inżynieria cech czasowych (godzina, dzień tygodnia, święta)
- [ ] Pobieranie i łączenie z danymi o cenach energii (RCE / TGE)
- [ ] Modele prognozowania szeregów czasowych (ARIMA/SARIMAX, Prophet, XGBoost/LightGBM, LSTM)

---

## 👤 Autor
**Marcin** — [@MarcinGoo](https://github.com/MarcinGoo)
