
#ticker_to_region.py

def ticker_to_region(ticker: str) -> str:
    parts = ticker.split(".")
    if len(parts) == 1:
        return "US"
    suffix = parts[-1].upper()
    region_map = {
        "AX": "AU",      
        "NZ": "NZ",
        "TO": "US",      
        "DE": "DE",
        "O":  "US",
    }
    return region_map.get(suffix, "US")



