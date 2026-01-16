# src/fmp_client.py
import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

FMP_API_KEY = os.getenv("FMP_API_KEY")

BASE_URL = "https://financialmodelingprep.com/stable"

class FMPError(Exception):
    pass

def _check_key():
    if not FMP_API_KEY:
        raise FMPError(
            "Missing FMP_API_KEY. Add it to your .env file like: FMP_API_KEY=xxxx"
        )

def fetch_prices(symbol: str) -> pd.DataFrame:
    _check_key()
    r = requests.get(
        f"{BASE_URL}/historical-price-eod/full",
        params={"symbol": symbol.upper(), "apikey": FMP_API_KEY},
        timeout=30
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty or "date" not in df.columns:
        raise FMPError(f"No price data returned for {symbol}.")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")[["date", "close"]].reset_index(drop=True)

def fetch_earnings(symbol: str) -> pd.DataFrame:
    _check_key()
    r = requests.get(
        f"{BASE_URL}/earnings",
        params={"symbol": symbol.upper(), "apikey": FMP_API_KEY},
        timeout=30
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty or "date" not in df.columns:
        raise FMPError(f"No earnings data returned for {symbol}.")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)
