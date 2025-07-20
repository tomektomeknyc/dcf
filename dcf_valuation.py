"""
DCF Valuation Module
Handles enterprise value calculations using different WACC methodologies
"""
import pandas as pd
import logging

# send INFO+ logs to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

import json
import os
from openai import OpenAI
from pathlib import Path
import streamlit as st

# Initialize OpenAI client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

@st.cache_data(show_spinner=True, max_entries=20)
def generate_fcff_projections(ticker, current_fcff, historical_data=None):
    logging.info(f"🔄 [FCFF] LLM projection for {ticker} (cache miss)")
    """
    Generate 5-year FCFF projections using LLM analysis
    
    Parameters:
    -----------
    ticker : str
        Company ticker symbol
    current_fcff : float
        Current year FCFF in millions
    historical_data : dict, optional
        Historical financial metrics for context
        
    Returns:
    --------
    list
        5-year FCFF projections [Year1, Year2, Year3, Year4, Year5]
    """
    try:
        prompt = f"""
        You are a financial analyst projecting Free Cash Flow to Firm (FCFF) for {ticker}.
        
        Current FCFF: ${current_fcff:.0f} million
        
        Please analyze this company and provide realistic 5-year FCFF projections considering:
        - Historical performance and growth trends
        - Industry outlook and competitive position
        - Capital allocation strategy and investment needs
        - Economic conditions and market dynamics
        
        Provide your analysis in JSON format with the following structure:
        {{
            "year_1_fcff": [amount in millions],
            "year_2_fcff": [amount in millions], 
            "year_3_fcff": [amount in millions],
            "year_4_fcff": [amount in millions],
            "year_5_fcff": [amount in millions],
            "reasoning": "Brief explanation of growth assumptions"
        }}
        
        Be realistic and conservative in your projections. Consider both growth opportunities and potential challenges.
        """
        
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a conservative financial analyst specializing in DCF valuations. Provide realistic cash flow projections based on company fundamentals."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=1000
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Extract actual LLM projections (no fallback multipliers)
        projections = [
            float(result["year_1_fcff"]),
            float(result["year_2_fcff"]),
            float(result["year_3_fcff"]),
            float(result["year_4_fcff"]),
            float(result["year_5_fcff"])
        ]
        
        print(f"LLM FCFF projections for {ticker}: {projections}")
        return projections
        
    except Exception as e:
        print(f"LLM FCFF projection failed for {ticker}: {e}")
        # Fallback to simple growth assumptions
        growth_rate = 0.05  # 5% annual growth
        return [current_fcff * (1 + growth_rate) ** i for i in range(1, 6)]


def get_net_debt(ticker, df_main):
    """
    Extract net debt from main dataset
    Net Debt = Total Debt - Cash and Cash Equivalents
    """
    try:
        # Get the most recent year data for this ticker (excluding Terminal Value)
        ticker_data = df_main[(df_main['Ticker'] == ticker) & (df_main['Year'] != 'Terminal Value')]
        if ticker_data.empty:
            return 0
        
        # Get the most recent year's data
        latest_year = ticker_data['Year'].max()
        latest_data = ticker_data[ticker_data['Year'] == latest_year].iloc[0]
        
        # Extract debt and cash from your existing data structure
        debt = latest_data.get('Debt', 0)
        cash = latest_data.get('Cash', 0)
        
        # Handle string values with commas
        if isinstance(debt, str):
            debt = float(debt.replace(',', '')) if debt != '-' else 0
        if isinstance(cash, str):
            cash = float(cash.replace(',', '')) if cash != '-' else 0
        
        # Ensure numeric values
        debt = float(debt) if pd.notna(debt) else 0
        cash = float(cash) if pd.notna(cash) else 0
        
        net_debt = debt - cash
        return net_debt
        
    except Exception as e:
        print(f"Net debt calculation failed for {ticker}: {e}")
        return 0


def calculate_intrinsic_value(ticker, wacc_method, fcff_table, wacc_df, df_main=None):
    """
    Calculate per-share intrinsic value using DCF methodology
    
    Parameters:
    -----------
    ticker : str
        Company ticker symbol
    wacc_method : str  
        WACC calculation method ('CAPM', 'FF5', or 'Damo')
    fcff_table : pd.DataFrame
        Table containing FCFF, Terminal Value, and Shares Outstanding data
    wacc_df : pd.DataFrame
        Table containing WACC rates for each method
    df_main : pd.DataFrame, optional
        Main dataset containing debt and cash data
        
    Returns:
    --------
    str
        Formatted intrinsic value per share or "-" if calculation fails
    """
    try:
        # Get Terminal Value and Shares Outstanding from fcff_table
        ticker_row = fcff_table[fcff_table["Ticker"] == ticker]
        if ticker_row.empty:
            return "-"
            
        tv_str = ticker_row["Terminal Value (M)"].iloc[0]
        shares_str = ticker_row["Shares Outstanding (M)"].iloc[0]
        
        if tv_str == "-" or shares_str == "-":
            return "-"
            
        terminal_value = float(tv_str)
        shares_outstanding = float(shares_str.replace(",", ""))
        
        # Get WACC rate for this ticker and method
        if ticker not in wacc_df.index:
            return "-"
            
        wacc_col = f"WACC ({wacc_method} %)"
        if wacc_col not in wacc_df.columns:
            return "-"
            
        wacc_pct = wacc_df.loc[ticker, wacc_col]
        if wacc_pct == "n/a" or pd.isna(wacc_pct):
            return "-"
            
        wacc_rate = float(wacc_pct) / 100  # Convert percentage to decimal
        
        # DCF Calculation Components:
        
        # 1. PV of Forecast FCFF (5-year present value using LLM projections)
        current_fcff_str = ticker_row["FCFF (Yr N)"].iloc[0]
        current_fcff = float(current_fcff_str.replace(",", ""))
        
        # Generate LLM-powered 5-year FCFF projections
        fcff_projections = generate_fcff_projections(ticker, current_fcff)
        
        # Calculate PV of each year's projected FCFF
        pv_fcff = sum(fcff_projections[i] / ((1 + wacc_rate) ** (i + 1)) for i in range(5))
        
        # 2. PV of Terminal Value (discounted back 5 years)
        pv_tv = terminal_value / ((1 + wacc_rate) ** 5)
        
        # 3. Enterprise Value = PV of FCFF + PV of Terminal Value
        enterprise_value = pv_fcff + pv_tv
        
        # 4. Equity Value = Enterprise Value - Net Debt
        # Net Debt = Total Debt - Cash and Cash Equivalents
        if df_main is not None:
            net_debt = get_net_debt(ticker, df_main)
        else:
            net_debt = 0  # Fallback if no main dataset
        equity_value = enterprise_value - net_debt
        
        # 5. Per-Share Intrinsic Value = Equity Value / Shares Outstanding
        intrinsic_value_per_share = equity_value / shares_outstanding
        
        return f"{intrinsic_value_per_share:.2f}"
        
    except (ValueError, IndexError, KeyError, ZeroDivisionError):
        return "-"


def calculate_all_intrinsic_values(fcff_table, wacc_df, df_main=None):
    """
    Calculate intrinsic values for all three WACC methods
    
    Parameters:
    -----------
    fcff_table : pd.DataFrame
        Table to update with intrinsic value calculations
    wacc_df : pd.DataFrame
        Table containing WACC rates
    df_main : pd.DataFrame, optional
        Main dataset containing debt and cash data for net debt calculation
        
    Returns:
    --------
    pd.DataFrame
        Updated fcff_table with intrinsic value columns populated
    """
    # Calculate intrinsic values for each method
    fcff_table["Intrinsic Value (CAPM)"] = fcff_table["Ticker"].apply(
        lambda ticker: calculate_intrinsic_value(ticker, "CAPM", fcff_table, wacc_df, df_main)
    )
    fcff_table["Intrinsic Value (FF5)"] = fcff_table["Ticker"].apply(
        lambda ticker: calculate_intrinsic_value(ticker, "FF5", fcff_table, wacc_df, df_main)
    )
    fcff_table["Intrinsic Value (Damo)"] = fcff_table["Ticker"].apply(
        lambda ticker: calculate_intrinsic_value(ticker, "Damo", fcff_table, wacc_df, df_main)
    )
    
    return fcff_table

