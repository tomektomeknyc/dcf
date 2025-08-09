# nwc.py
##### NWC START #####

"""
Compute sales, shares outstanding, and ΔNWC aligned to the app's `years`.
Depends only on `capm.grab_series`.
"""

from typing import Iterable, List, Optional, Tuple
import pandas as pd
import numpy as np

from capm import grab_series  # matches your file

def _to_series(x: Optional[Iterable], years: Iterable) -> pd.Series:
    """
    Coerce list/array/Series to a numeric Series aligned to `years` (as strings).
    If `x` is None, returns NaNs.
    """
    ys = [str(y) for y in years]
    if x is None:
        return pd.Series([np.nan] * len(ys), index=ys, dtype="float64")

    if isinstance(x, pd.Series):
        s = pd.to_numeric(x, errors="coerce").copy()
        # try to convert its index to year-like labels
        try:
            s.index = [str(pd.to_datetime(i).year) for i in s.index]
        except Exception:
            s.index = [str(i) for i in s.index]
        return s.reindex(ys)

    # list/tuple/ndarray
    s = pd.Series(list(x), dtype="float64")
    if len(s) == len(ys):
        s.index = ys
        return s
    # fallback: right-align to most recent years
    s = s.iloc[-len(ys):]
    s.index = ys[-len(s):]
    return s.reindex(ys)

# nwc.py
"""
Helpers to pull NWC data and compute ΔNWC in one go, using your exact block.
"""
from capm import grab_series

def compute_nwc_metrics(
    xlsx,             # Path or filename from app.py
    curr_assets,      # list[float] or None from grab_series
    curr_liab,        # list[float] or None from grab_series
    years             # list[int] of years
):
    """
    Returns:
      sales (list[float]),
      shares_outstanding (list[float]),
      change_in_nwc (list[float])
    """
    sales = grab_series(
        xlsx, "Income Statement", r"sales of goods & services\s*-\s*net"
    ) or [0.0] * len(years)

    shares_outstanding = grab_series(
        xlsx, "Balance Sheet", r"common shares.*outstanding.*total"
    ) or [0.0] * len(years)

    if curr_assets and curr_liab:
        nwc = [a - l for a, l in zip(curr_assets, curr_liab)]
        change_in_nwc = [0] + [nwc[i] - nwc[i - 1] for i in range(1, len(nwc))]
    else:
        change_in_nwc = [0] * len(years)

    return sales, shares_outstanding, change_in_nwc

# Convert outputs to plain lists in the same year order
    def _series_to_list(v):
        if isinstance(v, pd.Series):
            return [None if pd.isna(x) else float(x) for x in v.reindex(years_list)]
        return list(v) if isinstance(v, (list, tuple)) else [None] * len(years_list)

    sales_list  = _series_to_list(pd.Series(sales) if sales is not None else pd.Series([None]*len(years_list)))
    shares_list = _series_to_list(pd.Series(shares_outstanding) if shares_outstanding is not None else pd.Series([None]*len(years_list)))
    d_nwc_list  = [float(x) for x in d_nwc.reindex(years_list).fillna(0.0).tolist()]

    return sales_list, shares_list, d_nwc_list

__all__ = ["compute_nwc_metrics"]
##### NWC END #####