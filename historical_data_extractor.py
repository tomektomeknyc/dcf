# historical_data_extractor.py
import pandas as pd
from finance_data_loader import find_excel_file_for_ticker


def extract_all_sheets(ticker: str) -> dict:
    """
    Given a stock ticker, locate the Excel file and return all sheets as DataFrames.

    Args:
        ticker (str): Stock ticker (e.g. 'ADSK.O')

    Returns:
        dict[str, pd.DataFrame]: Dictionary of sheet names to DataFrames
    """
    excel_path = find_excel_file_for_ticker(ticker)

    
    try:
        xls = pd.ExcelFile(excel_path)
        sheets = {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}
        return sheets
    
    except Exception as e:
        raise RuntimeError(f"Failed to load Excel file for {ticker}: {e}")
