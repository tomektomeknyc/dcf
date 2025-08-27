# bridge_adapter.py
from excel_bridge import load_bridge_for
import yaml
import pandas as pd

def load_sources():
    with open("sources.yaml", "r") as f:
        return yaml.safe_load(f)["tickers"]

def load_inputs_from_bridge(ticker: str):
    sources = load_sources()
    b = load_bridge_for(ticker, sources)

    scal = b["scalars"]
    tbl  = b["tables"]["BR_FCFF_hist"].copy()

    # ensure numeric & sorted
    tbl["Year"] = pd.to_numeric(tbl["Year"])
    tbl["FCFF"] = pd.to_numeric(tbl["FCFF"])
    tbl = tbl.sort_values("Year")

    inputs = {
        "ticker": scal["BR_TICKER"],
        "currency": scal["BR_CURRENCY"],
        "rf": float(scal["BR_RF"]),
        "rd": float(scal["BR_RD"]),
        "tax": float(scal["BR_TAX"]),
        "wacc": float(scal["BR_WACC"]),
        "g_lt": float(scal["BR_G_LT"]),
        "shares": float(scal["BR_SHARES"]),
        "fcff_hist": tbl,  # DataFrame with Year, FCFF
        "source_type": b["type"],
        "source_path": b["path"],
    }
    return inputs
