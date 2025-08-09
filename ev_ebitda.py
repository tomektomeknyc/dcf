# ev_ebitda.py
"""
Compute Enterprise Value (EV) and EV/EBITDA ratio, along with top display metrics.
"""
import pandas as pd

##### EV/EBITDA START #####
def compute_ev_ebitda(
    sim: pd.DataFrame,
    ev_mult: float,
    hist_net_debt: float
) -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    """
    Given a simulated projections DataFrame `sim`, an EV multiple `ev_mult`,
    and the historical net debt `hist_net_debt`, computes:
      - sim['EV'] = sim['EBITDA'] * ev_mult + (sim_net_debt - hist_net_debt)
      - sim['EV/EBITDA'] = sim['EV'] / sim['EBITDA'], replacing zero EBITDA with NA.
    Also returns the list of metrics for display in your two-row panels.
    """
    # Recompute simulated net debt
    sim_net_debt = sim['Debt'] - sim['Cash']

    # EV and EV/EBITDA
    sim['EV'] = sim['EBITDA'] * ev_mult + (sim_net_debt - hist_net_debt)
    sim['EV/EBITDA'] = sim['EV'] / sim['EBITDA'].replace(0, pd.NA)

    # Top metrics: two-row panels, 5 columns each
    hist_metrics = [
        ("EBITDA",    "EBITDA",         "$ {:,.0f}"),
        ("CapEx",     "CapEx",          "$ {:,.0f}"),
        ("FCF",       "FCF",            "$ {:,.0f}"),
        ("EV",        "EV",             "$ {:,.0f}"),
        ("EV/EBITDA", "EV/EBITDA",      "{:.2f}x"),
        ("Debt",      "Debt",           "$ {:,.0f}"),
        ("Cash",      "Cash",           "$ {:,.0f}"),
        ("ΔNWC",      "ChangeNWC",      "$ {:,.0f}"),
        ("Interest",  "InterestExpense","$ {:,.0f}"),
        ("Tax Rate",  "tax_rate",       "{:.1%}"),
    ]
    return sim, hist_metrics
##### EV/EBITDA END #####

__all__ = ["compute_ev_ebitda"]