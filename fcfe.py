# fcfe.py

##### FCFE START #####
"""
Compute Free Cash Flow to Equity (FCFE).
"""

from typing import Union
import pandas as pd

def compute_fcfe(
    fcff: Union[pd.Series, list[float]],
    interest_expense: Union[pd.Series, list[float]],
    tax_rate: Union[pd.Series, list[float]],
    delta_debt: Union[pd.Series, list[float]],
    delta_cash: Union[pd.Series, list[float]]
) -> pd.Series:
    """
    FCFE = FCFF
         - InterestExpense * (1 - tax_rate)
         + ΔDebt
         - ΔCash

    Accepts either pd.Series or plain lists (will coerce to Series).
    Returns a pd.Series of FCFE.
    """
    # Coerce to Series if needed
    fcff_ie = pd.Series(fcff)
    ie      = pd.Series(interest_expense)
    tr      = pd.Series(tax_rate)
    dd      = pd.Series(delta_debt)
    dc      = pd.Series(delta_cash)

    return (
        fcff_ie
      - ie * (1 - tr)
      + dd
      - dc
    )

__all__ = ["compute_fcfe"]

##### FCFE END #####