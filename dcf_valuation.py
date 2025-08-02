#dcf_valuation.py
"""
DCF Valuation Module
Handles enterprise value calculations using different WACC methodologies
"""
import pandas as pd
import logging

# Send INFO+ logs to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

import json
import os
from openai import OpenAI
from pathlib import Path
import streamlit as st
# --- Global list for debugging (place at the top of dcf_valuation.py) ---

##### LLM START #####

all_debugs = {}  # Dict: {ticker: [line1, line2, line3]}

# Initialize OpenAI client


openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"].strip())



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
        
        
        # Determine growth tone for prompt based on ticker or industry
        tech_tickers = [
            "ADSK.O", "CSL.AX", "SEK.AX", "XRO.AX", "WTC.AX",  # Tech/Software stocks
            "INTU.O", "PYPL.O"  
        ]
        mature_tickers = [
            "DG", "HSY", "URI", "AIR.NZ", "FBU.NZ", "FCG.NZ", "MEL.NZ",  # Mature/industrial/airlines/slow-growth
            "BOSSn.DE", "DHLn.DE", "HFGG.DE", "KGX.DE", "SHLG.DE", "TMV.DE"
        ]

        if ticker in tech_tickers:
            tone = (
                "You are a financial analyst projecting Free Cash Flow to Firm (FCFF) for a **high-growth technology company**. "
                "Tech stocks like {ticker} should not be treated as mature or conservative firms—factor in sector leadership, rapid innovation, and the potential for accelerated revenue and FCF growth. "
                "Allow for ambitious (but not reckless) growth projections, and justify higher growth assumptions where appropriate for market leaders or companies launching major new products."
            )
        else:
            tone = (
                "You are a financial analyst projecting Free Cash Flow to Firm (FCFF) for a **mature or stable company**. "
                "Be prudent and use a conservative approach in estimating growth and discount rates, factoring in market share, economic cycles, and competitive pressures."
            )

        prompt = f"""
        {tone}

        Current FCFF: ${current_fcff:.0f} million

        Please analyze this company and provide realistic 5-year FCFF projections considering:
        - Historical performance and growth trends
        - Industry outlook and competitive position
        - Capital allocation strategy and investment needs
        - Economic conditions and market dynamics

        IMPORTANT:
        - When providing the perpetual growth rate (g) and discount rate (r) for DCF, you must ensure that r is at least 2 percentage points (0.02) higher than g. For example, if g is 0.03, r must be at least 0.05. Never set r ≤ g.
        - Explain in 1-2 sentences why your growth and discount rates are reasonable and confirm that r > g.

        Provide your analysis in JSON format with the following structure:
        {{
            "year_1_fcff": [amount in millions],
            "year_2_fcff": [amount in millions], 
            "year_3_fcff": [amount in millions],
            "year_4_fcff": [amount in millions],
            "year_5_fcff": [amount in millions],
            "growth_rate": [g as a decimal, e.g. 0.07],
            "discount_rate": [r as a decimal, e.g. 0.09],
            "reasoning": "Brief explanation of growth and discount rate assumptions"
        }}

        Respond ONLY with the JSON object.
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
        fcff_2030 = projections[4]  # 5th year in your LLM projections
       
        print(f"LLM FCFF projections for {ticker}: {projections}")
        # Store the message to st.session_state (or the cache)
        # Get growth and discount rates from the LLM response
        g = float(result.get("growth_rate", 0.03))    
        r = float(result.get("discount_rate", 0.10))  
        n = 4
        # ─── ENFORCE: r must be at least 2 percentage points above g ─────────────
        if r <= g + 0.02:
            print(f"⚠️ WARNING: LLM returned invalid r ({r}) <= g + 0.02 ({g + 0.02}) — applying fallback.")
            # You can choose your fallback. Example: set r = g + 0.03 (guaranteed safe margin)
            r = round(g + 0.03, 4)
            # Optionally, override the reasoning string so it’s clear what happened
            result["reasoning"] += " (Adjusted: Discount rate increased to maintain r > g + 0.02 for DCF realism.)"
        
            #print(f"DEBUG TV INPUTS: {ticker} | FCFF_2030={fcff_2030} | g={g} | r={r} | n={n}")

        TV = (fcff_2030  * (1+r))/ (r- g)
       
        PV_TV = TV / ((1 + r) ** n)
        tv_millions = PV_TV
        formatted_tv = f"{tv_millions:,.0f} M"
        
        msg = f"{ticker} - LLM FCFF projections (2026-2030):  {projections}  | growth rate (g): {g:.4f},  discount rate (r) : {r:.4f}, Terminal Value (2026) : {formatted_tv}"

        # Save the message in session state so app.py can access it
        
        if "llm_tv_debug_msgs" not in st.session_state:
            st.session_state["llm_tv_debug_msgs"] = {}
        st.session_state["llm_tv_debug_msgs"][ticker] = msg


        if "fcff_projection_cache" not in st.session_state:
            st.session_state["fcff_projection_cache"] = {}
        st.session_state["fcff_projection_cache"][ticker] = st.session_state["fcff_projection_cache"].get(ticker, {})
        st.session_state["fcff_projection_cache"][ticker]["msg"] = msg
        st.session_state["fcff_projection_cache"][ticker]["g"] = g
        st.session_state["fcff_projection_cache"][ticker]["r"] = r

        
        return projections, fcff_2030

        
    except Exception as e:
        print(f"LLM FCFF projection failed for {ticker}: {e}")
        # Fallback to simple growth assumptions
        growth_rate = 0.05  # 5% annual growth
        return [current_fcff * (1 + growth_rate) ** i for i in range(1, 6)]
##### LLM END #####    

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
        # ─── 1. Get Terminal Value and Shares Outstanding ──────────
        #ticker_row = fcff_table[fcff_table["Ticker"] == ticker]
        # Inside calculate_intrinsic_value
        ticker_row = fcff_table[fcff_table["Ticker"] == ticker].iloc[0]

        if ticker_row.empty:
            return "-"

        # 1. Get Shares Outstanding (from table)
     ######################################################
        shares_str = None
        if isinstance(fcff_table, pd.DataFrame):
            ticker_row = fcff_table[fcff_table["Ticker"] == ticker]
            if not ticker_row.empty:
                shares_str = ticker_row["Shares Outstanding (M)"].values[0]
        elif isinstance(fcff_table, dict) and "Shares Outstanding (M)" in fcff_table:
            shares_str = fcff_table["Shares Outstanding (M)"]
        elif isinstance(fcff_table, pd.Series) and "Shares Outstanding (M)" in fcff_table:
            shares_str = fcff_table["Shares Outstanding (M)"]

        if shares_str is None or shares_str == "-":
            return "-"
        shares_outstanding = float(str(shares_str).replace(",", ""))
     ########################################################

        # Get current FCFF (from table)
        current_fcff_str = ticker_row["FCFF (Yr N)"].iloc[0]
        current_fcff = float(str(current_fcff_str).replace(",", ""))

        # 2. Get FCFF projections, g, r (from LLM/session/UI)
        fcff_proj_list, _ = generate_fcff_projections(ticker, current_fcff)
      

       

        # ─── 2. Get WACC rate for this ticker/method ──────────────
        wacc_col = f"WACC ({wacc_method} %)"
        if wacc_col not in wacc_df.columns:
            return "-"
        if ticker not in wacc_df.index:
            return "-"
        wacc_pct = wacc_df.loc[ticker, wacc_col]
        if wacc_pct == "n/a" or pd.isna(wacc_pct):
            return "-"
        wacc_rate = float(wacc_pct) / 100  # convert to decimal

        # ─── 3. Get FCFF for year N (current year) ────────────────
        current_fcff_str = ticker_row["FCFF (Yr N)"].iloc[0]
        current_fcff = float(str(current_fcff_str).replace(",", ""))

        # ─── 4. Generate 5-year FCFF projections ──────────────────
        fcff_proj_list, fcff_2030 = generate_fcff_projections(ticker, current_fcff)

        # ─── 5. Calculate PV of each year's projected FCFF ────────
        pv_fcff = sum(
            fcff_proj_list[i] / ((1 + wacc_rate) ** (i + 1))
            for i in range(5)
        )

     
        # Always use last projected FCFF for TV with latest g/r from cache/session/LLM/UI
        fcff_last = fcff_proj_list[-1]
        if "fcff_projection_cache" in st.session_state and ticker in st.session_state["fcff_projection_cache"]:
            g = st.session_state["fcff_projection_cache"][ticker].get("g", 0.03)
            r = st.session_state["fcff_projection_cache"][ticker].get("r", wacc_rate)
        else:
            g = 0.03
            r = wacc_rate

        tv_calc = fcff_last * (1 + g) / (r - g) if r > g else 0
        pv_tv = tv_calc / ((1 + r) ** 5)


                # ─── 7. Enterprise Value ──────────────────────────────────
        enterprise_value = pv_fcff + pv_tv

        # ─── 8. Net Debt ─────────────────────────────────────────
        if df_main is not None:
            net_debt = get_net_debt(ticker, df_main)
        else:
            net_debt = 0  # fallback

        # ─── 9. Equity Value and Per-share ───────────────────────
        equity_value = enterprise_value - net_debt
        intrinsic_value_per_share = equity_value / shares_outstanding

        # ── 10. DEBUG LOGGING ───────────────────────────────────────────────
        global all_debugs
        if ticker not in all_debugs:
            all_debugs[ticker] = {}
        debug_text = (
            f"{ticker} [{wacc_method}]: "
            f"equity_value = {equity_value}, "
            f"shares_outstanding = {shares_outstanding}, "
            f"intrinsic_value_per_share = {intrinsic_value_per_share}"
        )
        all_debugs[ticker][wacc_method] = debug_text



        return f"{intrinsic_value_per_share:.2f}"

    except (ValueError, IndexError, KeyError, ZeroDivisionError) as e:
        print(f"Error in calculate_intrinsic_value: {e}")
        return "-"




def calculate_all_intrinsic_values(fcff_table, wacc_df, df_main=None):
    """
    Calculate intrinsic values for all three WACC methods.
    ALWAYS use a fresh DataFrame for calculations.
    
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

    # --- Create a COPY to ensure we're not mutating an old table in place ---
    fcff_table = fcff_table.copy()

    # Remove any old intrinsic value columns before re-calculating
    for col in ["Intrinsic Value (CAPM)", "Intrinsic Value (FF5)", "Intrinsic Value (Damo)"]:
        if col in fcff_table.columns:
            fcff_table.drop(columns=col, inplace=True)

    # Calculate intrinsic values for each method from scratch
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


