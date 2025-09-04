# data_gatekeeper.py — minimal, returns-only module
# Purpose:
#   Validate and standardize STOCK and INDEX returns files from attached_assets/.
#   Returns a DataFrame with exactly two columns: ["Date", "MarketReturn"].
#
# Accepted inputs
#   • CSV, XLSX, XLS
#   • Stock returns:
#       - ["Date","Return"]           → renamed to MarketReturn
#       - ["Date","MarketReturn"]     → used as-is
#       - ["Date","CLOSE"]            → MarketReturn = pct_change(CLOSE)
#   • Index (market) returns:
#       - ["Date","MarketReturn"]     → used as-is
#       - ["Date","Return"]           → renamed to MarketReturn
#
# Not included (by design):
#   • OCR, PDFs, image handling, generic loaders (kept out to keep this lean)

import os
import pandas as pd


REQUIRED_RET_COLS = ["Date", "MarketReturn"]


def _read_returns_structured(path_str: str) -> pd.DataFrame:
    """
    Read CSV/XLS/XLSX as a DataFrame.
    Keeps 'Date' as a column (repairs if it came in as the index).
    Raises ValueError on unsupported extension.
    """
    ext = os.path.splitext(str(path_str))[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path_str)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path_str)
    else:
        raise ValueError(
            f"Unsupported file type for returns: '{ext}'. Use CSV/XLSX/XLS."
        )

    # Ensure 'Date' is a column, not only an index
    if "Date" not in df.columns:
        # If index name is 'Date', reset it
        if getattr(df.index, "name", None) and str(df.index.name).lower() == "date":
            df = df.reset_index()
        # If index is DatetimeIndex, also reset to a column named 'Date'
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "Date"})

    return df


def _standardize_returns_df(
    df: pd.DataFrame,
    *,
    allow_close_to_compute: bool,
) -> pd.DataFrame | None:
    """
    Normalize to exactly ['Date','MarketReturn']:
      1) Parse 'Date'
      2) Prefer 'MarketReturn'; else rename 'Return'→'MarketReturn'
      3) If still missing and allowed, compute from 'CLOSE' via pct_change()
      4) Coerce 'MarketReturn' to numeric and drop NaNs
      5) Sort by 'Date' and reset index
    """
    if "Date" not in df.columns:
        return None

    # Parse dates
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=False)
    if df["Date"].isna().all():
        return None

    # Standardize the return column name
    lower = {c.lower(): c for c in df.columns}
    mr_col = None
    if "marketreturn" in lower:
        mr_col = lower["marketreturn"]
    elif "return" in lower:
        mr_col = lower["return"]
        df = df.rename(columns={mr_col: "MarketReturn"})
        mr_col = "MarketReturn"

    # If still missing and allowed, compute from CLOSE
    if mr_col is None and allow_close_to_compute and "CLOSE" in df.columns:
        df = df.sort_values("Date")
        df["MarketReturn"] = pd.to_numeric(df["CLOSE"], errors="coerce").pct_change()
        mr_col = "MarketReturn"

    if mr_col is None:
        return None

    # Coerce to numeric & clean rows
    df["MarketReturn"] = pd.to_numeric(df[mr_col], errors="coerce")
    df = df.dropna(subset=["MarketReturn"])
    if df.empty:
        return None

    # Final tidy: keep only required columns, sorted
    out = df[["Date", "MarketReturn"]].copy()
    out = out.sort_values("Date").reset_index(drop=True)
    return out


def load_stock_returns(path_like) -> pd.DataFrame:
    """
    Validate a *stock* returns file (attached_assets/returns_<TICKER>.csv).

    Accepts:
      - Date, Return
      - Date, MarketReturn
      - Date, CLOSE  (we compute pct_change)
    Returns:
      DataFrame with columns ['Date', 'MarketReturn'].
    Raises:
      ValueError on invalid format/content.
    """
    path_str = os.fspath(path_like)
    df_raw = _read_returns_structured(path_str)
    df_std = _standardize_returns_df(df_raw, allow_close_to_compute=True)
    if df_std is None:
        raise ValueError(
            "Invalid stock returns file. Expected 'Date' plus one of: "
            "'Return' or 'MarketReturn', or 'CLOSE' so returns can be computed."
        )
    return df_std


def load_index_returns(path_like) -> pd.DataFrame:
    """
    Validate an *index* market returns file (attached_assets/<INDEX>_market_returns.csv).

    Accepts:
      - Date, MarketReturn
      - Date, Return  (renamed to MarketReturn)
    Returns:
      DataFrame with columns ['Date', 'MarketReturn'].
    Raises:
      ValueError on invalid format/content.
    """
    path_str = os.fspath(path_like)
    df_raw = _read_returns_structured(path_str)
    df_std = _standardize_returns_df(df_raw, allow_close_to_compute=False)
    if df_std is None:
        raise ValueError(
            "Invalid market returns file. Expected 'Date' and 'MarketReturn' (or 'Return')."
        )
    return df_std
