##### RISK HEDGE START #####
# risk_hedge.py
# Clean utilities for hedge variance and optimal hedge ratio.
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class HedgeStats:
    sigma_S: float        # spot std dev
    sigma_F: float        # futures/proxy std dev
    rho: float            # correlation between spot and futures
    h_star: float         # optimal hedge ratio
    var_min: float        # variance at h*
    var_unhedged: float   # variance at h=0 (baseline)

def _align(spot: pd.Series, fut: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Align series on common index and drop NaNs."""
    df = pd.concat([spot.rename("S"), fut.rename("F")], axis=1).dropna()
    return df["S"], df["F"]

def compute_hedge_stats(spot_returns: pd.Series, futures_returns: pd.Series) -> HedgeStats:
    """
    Compute σ_S, σ_F, ρ, optimal hedge ratio h*, baseline variance, and min variance.
    Inputs are returns (simple or log), same frequency, pandas Series indexed by date.
    """
    S, F = _align(spot_returns, futures_returns)

    if len(S) < 2:
        raise ValueError("Need at least 2 overlapping return observations.")

    sigma_S = float(S.std(ddof=1))
    sigma_F = float(F.std(ddof=1))
    rho = float(S.corr(F))

    if sigma_F == 0.0:
        raise ValueError("Futures/proxy variance is zero; cannot compute h*.")

    # Power Rule used implicitly in the derivation of this closed form:
    # V(h) = σ_S^2 - 2 h ρ σ_S σ_F + h^2 σ_F^2  →  dV/dh = -2 ρ σ_S σ_F + 2 h σ_F^2
    # Set to 0 → h* = (ρ σ_S) / σ_F
    h_star = (rho * sigma_S) / sigma_F

    var_unhedged = sigma_S ** 2
    var_min = (sigma_S ** 2) - 2*h_star*rho*sigma_S*sigma_F + (h_star ** 2) * (sigma_F ** 2)

    return HedgeStats(
        sigma_S=sigma_S,
        sigma_F=sigma_F,
        rho=rho,
        h_star=h_star,
        var_min=var_min,
        var_unhedged=var_unhedged,
    )

def hedge_variance_curve(spot_returns: pd.Series, futures_returns: pd.Series,
                         h_values: np.ndarray) -> pd.DataFrame:
    """
    Build V(h) across a grid of hedge ratios for plotting.
    Returns DataFrame with columns: 'h', 'V(h)'.
    """
    S, F = _align(spot_returns, futures_returns)
    sigma_S = float(S.std(ddof=1))
    sigma_F = float(F.std(ddof=1))
    rho = float(S.corr(F))

    Vh = (sigma_S**2) - 2*h_values*rho*sigma_S*sigma_F + (h_values**2)*(sigma_F**2)
    return pd.DataFrame({"h": h_values, "V(h)": Vh})
##### RISK HEDGE END #####
