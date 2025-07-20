# stock_returns.py

import refinitiv.dataplatform.eikon as ek
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import os

# ─── Load API key from .env ──────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("REFINITIV_APP_KEY")
if not API_KEY:
    raise EnvironmentError("REFINITIV_APP_KEY not found in .env")

ek.set_app_key(API_KEY)

# ─── Fetch Daily Returns ──────────────────────────────────────────────────────
def fetch_daily_returns(ticker: str, years: int = 10) -> pd.DataFrame:
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=365 * years)

    df = ek.get_timeseries(
        rics=ticker,
        fields="CLOSE",
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        interval="daily"
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}")

    df["Return"] = df["CLOSE"].pct_change()
    df.dropna(subset=["Return"], inplace=True)
    return df

# ─── Save Stock Returns to CSV ────────────────────────────────────────────────
def save_returns_to_csv(ticker: str, df: pd.DataFrame):
    Path("attached_assets").mkdir(exist_ok=True)
    path = Path("attached_assets") / f"returns_{ticker.replace('.', '_')}.csv"
    df.to_csv(path)
    print(f"✅ Saved returns to: {path}")
    return path

# ─── Fetch Daily Market Index Returns ─────────────────────────────────────────
def fetch_market_returns(index_ric: str, years: int = 10) -> pd.DataFrame:
    """
    index_ric : the RIC for your index, e.g. ".SPX" or ".AXJO"
    years     : lookback window in years
    """
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=365 * years)

    df = ek.get_timeseries(
        rics=index_ric,
        fields="CLOSE",
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        interval="daily"
    )
    if df is None or df.empty:
        raise ValueError(f"No market data for {index_ric}")

    df["MarketReturn"] = df["CLOSE"].pct_change()
    df.dropna(subset=["MarketReturn"], inplace=True)
    return df[["MarketReturn"]]

# ─── Save Market Returns to CSV ───────────────────────────────────────────────
def save_market_returns(index_ric: str, df: pd.DataFrame):
    Path("attached_assets").mkdir(exist_ok=True)
    name = index_ric.lstrip(".").replace(".", "_")
    path = Path("attached_assets") / f"{name}_market_returns.csv"
    df.to_csv(path)
    print(f"✅ Saved market returns to: {path}")
    return path

# ─── Main Function to Loop Through Tickers & Regions ─────────────────────────
def main():
    # 1) Fetch stock returns
    tickers = [
        "CSL.AX", "FLT.AX", "SEK.AX", "WTC.AX", "XRO.AX",
        "BOSSn.DE","DHLn.DE","HFGG.DE","KGX.DE","SHLG.DE",
        "TMV.DE","AIR.NZ","FBU.NZ","FCG.NZ","MEL.NZ",
        "ADSK.O","DG","HSY","INTU.O","PYPL.O","URI"
    ]
    for ticker in tickers:
        try:
            df = fetch_daily_returns(ticker)
            save_returns_to_csv(ticker, df)
        except Exception as e:
            print(f"❌ Failed for {ticker}: {e}")

    # 2) Fetch each region’s benchmark index returns
    region_indices = {
        "US": ".SPX",      # S&P 500
        "AU": ".AXJO",   # ASX 200 TR
        "NZ": ".NZ50",     # NZX 50 TR
        "DE": ".GDAXI",    # DAX
    }
    for region, ric in region_indices.items():
        try:
            mkt_df = fetch_market_returns(ric)
            save_market_returns(ric, mkt_df)
        except Exception as e:
            print(f"❌ Failed to fetch {region} market ({ric}): {e}")

# ─── Run If Standalone ────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
