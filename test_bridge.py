# test_bridge.py
from excel_bridge import load_bridge_for
import yaml

with open("sources.yaml", "r") as f:
    sources = yaml.safe_load(f)["tickers"]

for t in ["TEREPAY", "SUPAHUMAN"]:
    bridge = load_bridge_for(t, sources)
    print("\n---", t, "---")
    print("scalars:", bridge["scalars"])
    print("first rows of BR_FCFF_hist:")
    print(bridge["tables"]["BR_FCFF_hist"].head())
