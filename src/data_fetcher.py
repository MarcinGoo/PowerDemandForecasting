import argparse
import time
import urllib3
import requests
import pandas as pd

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PSE_API_URL = "https://api.raporty.pse.pl/api/his-wlk-cal"


def fetch_pse_history(max_days: int = 30, verbose: bool = True) -> pd.DataFrame:
    """
    Fetches data from the PSE API page by page (using the nextLink parameter).

    Args:
        max_days (int): Maximum number of pages/days to fetch.
        verbose (bool): Whether to display progress logs.

    Returns:
        pd.DataFrame: Ordered and cleaned DataFrame with energy data.
    """
    url = PSE_API_URL
    headers = {"User-Agent": "Mozilla/5.0"}
    all_records = []
    fetched_pages = 0

    if verbose:
        print(f"Started fetching data from PSE (maximum {max_days} days/pages)...")

    while url and fetched_pages < max_days:
        try:
            response = requests.get(url, headers=headers, verify=False, timeout=15)
        except requests.RequestException as e:
            if verbose:
                print(f"Connection error: {e}")
            break

        if response.status_code != 200:
            if verbose:
                print(f"HTTP Error {response.status_code}")
            break

        data = response.json()
        records = data.get("value", [])
        all_records.extend(records)

        url = data.get("nextLink")
        fetched_pages += 1

        if verbose and fetched_pages % 5 == 0:
            print(f"Fetched {fetched_pages} pages ({len(all_records)} rows)...")

        time.sleep(0.1)  # Short pause to prevent rate limiting

    if verbose:
        print(f"Fetching completed! Total records fetched: {len(all_records)}.")

    if not all_records:
        return pd.DataFrame()

    # Create and clean DataFrame
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

    # Feature engineering for RES and net demand
    if "pv" in df.columns and "wi" in df.columns:
        df["oze_total"] = df["pv"] + df["wi"]
        if "demand" in df.columns:
            df["net_demand"] = df["demand"] - df["oze_total"]

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetching data from PSE API")
    parser.add_argument("--days", type=int, default=30, help="Number of days/pages to fetch (default: 30)")
    parser.add_argument("--output", type=str, default="data/raw/pse_energy_data.csv", help="CSV file save path")
    args = parser.parse_args()

    df_result = fetch_pse_history(max_days=args.days)
    if not df_result.empty:
        df_result.to_csv(args.output, index=False)
        print(f"Successfully saved {len(df_result)} records to '{args.output}'")
    else:
        print("No data fetched.")
