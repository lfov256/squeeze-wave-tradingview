#!/usr/bin/env python3
"""
Script robusto para descargar máximo histórico de Massive/Polygon
en formato Parquet con soporte incremental y reintentos.
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path("data/historical")
DATA_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "XAUUSD": "C:XAUUSD",
    "BTCUSD": "X:BTCUSD",
    "ETHUSD": "X:ETHUSD",
    "EURUSD": "C:EURUSD",
}

API_KEY = os.getenv("POLYGON_API_KEY")

HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_with_retry(url, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            if response.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"Rate limit hit. Waiting {wait}s...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(5 * (attempt + 1))
    raise Exception(f"Failed to fetch after {max_retries} attempts")


def download_ticker(ticker: str, name: str):
    filepath = DATA_DIR / f"{name}.parquet"

    # Si ya existe, descargar solo datos nuevos
    if filepath.exists():
        existing_df = pd.read_parquet(filepath)
        last_date = existing_df["Date"].max()
        start_date = last_date + timedelta(days=1)
        print(f"Updating {name} from {start_date}...")
    else:
        start_date = datetime(2000, 1, 1).date()  # Máximo histórico posible
        print(f"Downloading full history for {name}...")

    end_date = datetime.now().date()

    if start_date > end_date:
        print(f"{name} is already up to date.")
        return

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
        f"{start_date}/{end_date}?adjusted=true&sort=asc&limit=50000&apiKey={API_KEY}"
    )

    try:
        data = fetch_with_retry(url)
        if "results" not in data or not data["results"]:
            print(f"No new data for {name}")
            return

        df = pd.DataFrame(data["results"])
        df["Date"] = pd.to_datetime(df["t"], unit="ms").dt.date
        df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

        if filepath.exists():
            old_df = pd.read_parquet(filepath)
            df = pd.concat([old_df, df]).drop_duplicates(subset=["Date"]).sort_values("Date")

        df.to_parquet(filepath, index=False)
        print(f"Saved {len(df)} rows for {name}")

    except Exception as e:
        print(f"Error downloading {name}: {e}")


def main():
    if not API_KEY:
        print("ERROR: POLYGON_API_KEY not set")
        return

    for name, ticker in ASSETS.items():
        download_ticker(ticker, name)
        time.sleep(2)  # Pequeña pausa entre tickers

if __name__ == "__main__":
    main()