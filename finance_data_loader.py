#finance_data_loader.py
import pandas as pd
from pathlib import Path

# Define base paths for financial data by region
EXCEL_FOLDERS = {
    "AU": Path("AU"),
    "NZ": Path("NZ"),
    "DE": Path("DE"),
    "US": Path("US"),
}


# STEP 2: Define utility function to locate Excel file for a given ticker

def find_excel_file_for_ticker(ticker):
    """
    Search all known region folders for an Excel file matching the ticker.
    Returns the full path to the Excel file if found, else None.
    """
    for region, folder in EXCEL_FOLDERS.items():
        excel_path = folder / f"{ticker}.xlsx"
        if excel_path.exists():
            return excel_path
    return None


# STEP 3: Load and return all sheets from the Excel file

def load_financial_sheets(ticker):
    """
    Given a stock ticker, locate the matching Excel file and return
    a dictionary of all DataFrames in the workbook, keyed by sheet name.
    """
    excel_file = find_excel_file_for_ticker(ticker)
    if excel_file is None:
        raise FileNotFoundError(f"No Excel file found for ticker: {ticker}")

    xls = pd.ExcelFile(excel_file)
    return {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}
