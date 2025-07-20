import pandas as pd
import numpy as np

def compute_driver_metrics(df: pd.DataFrame, lookback: int = 5) -> dict[str, dict[str, float]]:
    """
    For each ticker, compute:
    - ebit_cagr
    - avg_tax_rate
    - da_growth
    - capex_growth
    - avg_dNWC_pct_sales
    over the last `lookback` years in df.
    """
    drivers = {}
    
    # Get the actual latest year from the data (don't assume 2025)
    available_years = df[df["Year"] != "Terminal Value"]["Year"].dropna()
    if available_years.empty:
        return drivers
        
    latest_year = int(available_years.max())
    start_year = latest_year - lookback

    for ticker, sub in df.groupby("Ticker"):
        # Filter to only numeric years and sort
        numeric_data = sub[sub["Year"] != "Terminal Value"].copy()
        numeric_data["Year"] = pd.to_numeric(numeric_data["Year"], errors='coerce')
        numeric_data = numeric_data.dropna(subset=["Year"])
        
        if numeric_data.empty:
            continue
            
        hist = numeric_data.set_index("Year").sort_index()
        
        # Get available years within our lookback period
        available_years_in_range = hist.index[(hist.index >= start_year) & (hist.index <= latest_year)]
        
        if len(available_years_in_range) < 2:  # Need at least 2 years for calculations
            continue
            
        # Use only the available years
        hist = hist.loc[available_years_in_range]
        
        # Safe function to get first and last values
        def safe_cagr(series, years_span):
            if len(series) < 2:
                return 0.0
            first_val = series.iloc[0]
            last_val = series.iloc[-1]
            if first_val > 0 and last_val > 0 and years_span > 0:
                return (last_val / first_val) ** (1/years_span) - 1
            return 0.0
        
        years_span = len(hist) - 1
        
        # 1) EBIT CAGR
        ebit_cagr = safe_cagr(hist["EBIT"].dropna(), years_span)

        # 2) Avg tax rate (using CashTaxesPaid / EBIT)
        sum_tax = hist["CashTaxesPaid"].sum()
        sum_ebit = hist["EBIT"].sum()
        avg_tax_rate = sum_tax / sum_ebit if sum_ebit > 0 else 0.0

        # 3) D&A growth
        da_growth = safe_cagr(hist["DepreciationAndAmortization"].dropna(), years_span)

        # 4) CapEx growth (taking absolute values since CapEx is usually negative)
        capex_series = hist["CapEx"].dropna().abs()
        capex_growth = safe_cagr(capex_series, years_span)

        # 5) Avg ΔNWC / Sales
        if "Sales" in hist.columns:
            dNWC_pct = (hist["ChangeNWC"] / hist["Sales"]).replace([np.inf, -np.inf], np.nan)
            avg_dNWC_pct_sales = float(dNWC_pct.mean(skipna=True))
        else:
            avg_dNWC_pct_sales = 0.0

        drivers[ticker] = {
            "ebit_cagr": ebit_cagr,
            "avg_tax_rate": avg_tax_rate,
            "da_growth": da_growth,
            "capex_growth": capex_growth,
            "avg_dNWC_pct_sales": avg_dNWC_pct_sales,
        }

    return drivers