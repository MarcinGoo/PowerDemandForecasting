import argparse
import time
import urllib3
import requests
import pandas as pd

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PSE_API_URL = "https://api.raporty.pse.pl/api/his-wlk-cal"


def fetch_pse_history(max_days: int = 30, verbose: bool = True) -> pd.DataFrame:
    """
    Pobiera dane z API PSE strona po stronie (za pomoca parametru nextLink).

    Args:
        max_days (int): Maksymalna liczba stron/dni do pobrania.
        verbose (bool): Czy wyswietlac logi postepu.

    Returns:
        pd.DataFrame: Uporzadkowany i oczyszczony DataFrame z danymi energetycznymi.
    """
    url = PSE_API_URL
    headers = {"User-Agent": "Mozilla/5.0"}
    all_records = []
    fetched_pages = 0

    if verbose:
        print(f"Rozpoczeto pobieranie danych z PSE (maksymalnie {max_days} dni/stron)...")

    while url and fetched_pages < max_days:
        try:
            response = requests.get(url, headers=headers, verify=False, timeout=15)
        except requests.RequestException as e:
            if verbose:
                print(f"Blad polaczenia: {e}")
            break

        if response.status_code != 200:
            if verbose:
                print(f"Blad HTTP {response.status_code}")
            break

        data = response.json()
        records = data.get("value", [])
        all_records.extend(records)

        url = data.get("nextLink")
        fetched_pages += 1

        if verbose and fetched_pages % 5 == 0:
            print(f"Pobrano {fetched_pages} stron ({len(all_records)} wierszy)...")

        time.sleep(0.1)  # Krotka pauza zapobiegajaca limitom zadan

    if verbose:
        print(f"Zakonczono pobieranie! Lacznie pobrano {len(all_records)} rekordow.")

    if not all_records:
        return pd.DataFrame()

    # Tworzenie i czyszczenie DataFrame
    df = pd.DataFrame(all_records)
    selected_columns = ["dtime", "demand", "pv", "wi", "jg"]
    existing_cols = [c for c in selected_columns if c in df.columns]
    df = df[existing_cols].copy()

    if "dtime" in df.columns:
        df["dtime"] = pd.to_datetime(df["dtime"])

    numeric_cols = ["demand", "pv", "wi", "jg"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("dtime").reset_index(drop=True)

    # Inzynieria cech OZE i zapotrzebowania netto
    if "pv" in df.columns and "wi" in df.columns:
        df["oze_total"] = df["pv"] + df["wi"]
        if "demand" in df.columns:
            df["net_demand"] = df["demand"] - df["oze_total"]

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pobieranie danych z API PSE")
    parser.add_argument("--days", type=int, default=30, help="Liczba dni/stron do pobrania (domyslnie: 30)")
    parser.add_argument("--output", type=str, default="data/raw/dane_energetyczne_pse.csv", help="Sciezka zapisu pliku CSV")
    args = parser.parse_args()

    df_result = fetch_pse_history(max_days=args.days)
    if not df_result.empty:
        df_result.to_csv(args.output, index=False)
        print(f"Pomyslnie zapisano {len(df_result)} rekordow do '{args.output}'")
    else:
        print("Nie pobrano zadnych danych.")
