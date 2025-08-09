#capm.py

##### CAPM START #####
# """
# Contains helper functions to compute CAPM betas and related measures.
# """
import streamlit as st
from scrape_ff5 import get_ff5_data_by_folder
import pandas as pd
from pathlib import Path

# Constants for Excel layout (Excel row index and column indices for data)
YEAR_ROW = 10
COLS     = list(range(1, 16))


def compute_pure_capm_beta(stock_ret: pd.Series, market_ret: pd.Series) -> float:
    """
    Compute β = Cov(Stock, Market) / Var(Market)
    on their overlapping dates.
    """
    df_combined = pd.concat(
        [stock_ret.rename("stock"), market_ret.rename("market")],
        axis=1,
    ).dropna()
    cov_sm = df_combined["stock"].cov(df_combined["market"])
    var_m  = df_combined["market"].var()
    return cov_sm / var_m if var_m else 0.0


def load_sheet(xlsx: Path, sheet: str) -> tuple[pd.DataFrame | None, list[int] | None]:
    """
    Load a raw Excel sheet without headers and extract the year labels.

    Returns:
        df: the full DataFrame or None if shape is unexpected
        years: list of integer years from the specified header row
    """
    try:
        df = pd.read_excel(xlsx, sheet_name=sheet, header=None, engine="openpyxl")
    except Exception:
        return None, None
    if df.shape[0] <= YEAR_ROW or df.shape[1] <= max(COLS):
        return None, None
    years = df.iloc[YEAR_ROW, COLS].astype(int).tolist()
    return df, years


def grab_series(xlsx: Path, sheet: str, regex: str) -> list[float] | None:
    """
    Grab the first row matching a regex in the first column and return numeric values
    for the configured columns. Returns None if not found.
    """
    df, years = load_sheet(xlsx, sheet)
    if df is None:
        return None
    col0 = df.iloc[:, 0].astype(str).str.lower()
    mask = col0.str.contains(regex, regex=True, na=False)
    if not mask.any():
        return None
    row = df.loc[mask, :].iloc[0]
    return pd.to_numeric(row.iloc[COLS], errors="coerce").tolist()



__all__ = [
    "compute_pure_capm_beta",
    "load_sheet",
    "grab_series",
]
##### CAPM END #####