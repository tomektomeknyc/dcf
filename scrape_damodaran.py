# scrape_damodaran.py
import requests
import pandas as pd
from bs4 import BeautifulSoup

DAMODARAN_URL = (
    "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/Betas.html"
)

def fetch_damodaran_betas() -> pd.DataFrame:
    """Download and parse Damodaran's US industry levered betas."""
    resp = requests.get(DAMODARAN_URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # The table is the first one on the page
    table = soup.find("table")
    df = pd.read_html(str(table))[0]

    # Clean column names (e.g. remove spaces/newlines)
    df.columns = [c.strip().replace("\n", " ") for c in df.columns]
    # Optional: rename columns to simpler keys
    df.rename(columns={
        "Industry": "Industry",
        "Levered Beta": "LeveredBeta",
        "Unlevered Beta": "UnleveredBeta",
        # ...add any others you need
    }, inplace=True)

    return df

if __name__ == "__main__":
    betas_df = fetch_damodaran_betas()
    print(betas_df.head())
