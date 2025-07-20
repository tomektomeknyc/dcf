# fcf_calculations.py
import pandas as pd
import streamlit as st
import os
from openai import OpenAI

# ─── LLM REAL CLIENT ──────────────────────────────────────────────────────────────


client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"].strip())



##### LLM START #####
def call_llm(prompt: str) -> str:
    """
    Send prompt to OpenAI and return the raw text response.
    """
    response = client.chat.completions.create(
        model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()

def extract_number(text: str, field: str) -> float:
    """
    Extract a percentage from LLM output.
    Returns the first number before a '%' sign.
    """
    import re
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\%", text)
    return float(m.group(1)) / 100 if m else 0.0

def compute_fcff(
    ticker: str,
    df_is: pd.DataFrame,
    df_cf: pd.DataFrame,
    df_bs: pd.DataFrame,
    year_col: str = "Year",
) -> pd.Series:
    """
    Compute historical Free Cash Flow to Firm (FCFF) for a given ticker.
    """
    # ─── Pull in your historical driver metrics ─────────────────────────────────
    drivers = st.session_state["drivers"]    # dict: ticker → metrics
    my      = drivers[ticker]                # e.g. {"ebit_cagr":0.12, ...}
    
    # ─── Pull in industry & region for better LLM context ────────────────────
    industry = st.session_state["industry_map"].get(ticker, "General")
    region   = st.session_state["region_map"   ].get(ticker, "Unknown")
    
    # ─── Build the prompt for the LLM ───────────────────────────────────────────
    prompt = f"""
    Here are key historical drivers for {ticker} (last 5 years):
    • EBIT CAGR:       {my['ebit_cagr']:.1%}
    • Avg tax rate:    {my['avg_tax_rate']:.1%}
    • D&A growth:      {my['da_growth']:.1%}
    • CapEx growth:    {my['capex_growth']:.1%}
    • ΔNWC/Sales:      {my['avg_dNWC_pct_sales']:.1%}

    Based on these, suggest for FY2026:
    1. A reasonable long-term growth rate (g)  
    2. A suitable discount rate (r)
    """
    
    # Get LLM response and extract g and r
    llm_resp = call_llm(prompt)  
    g = extract_number(llm_resp, "growth")       # e.g. 0.03 for 3%
    r = extract_number(llm_resp, "discount rate")# e.g. 0.10 for 10%
    
    # Fallback values if LLM extraction fails
    if g <= 0 or g > 0.1:  # Reasonable bounds for growth rate
        g = 0.03
    if r <= 0 or r > 0.3:  # Reasonable bounds for discount rate
        r = 0.10
    
    # ─── Extract data for FCFF calculation ─────────────────────────────────────
    # Get the most recent year's data from df_is
    if df_is.empty:
        return pd.Series(dtype=float, name='FCFF')
    
    # Get historical FCFF from the main dataframe (already calculated)
    fcff_historical = df_is["FCFF"] if "FCFF" in df_is.columns else pd.Series(dtype=float)
    
    if fcff_historical.empty:
        return pd.Series(dtype=float, name='FCFF')
    
    # Get the last year's FCFF value
    last_fcff = fcff_historical.iloc[-1]
    
    # Calculate FCFF₂₆ = FCFF₂₅ × (1 + g) using LLM growth rate
    fcff_2026 = last_fcff * (1 + g)
    
    # Calculate Terminal Value using Gordon Growth Model
    terminal_value = fcff_2026 * (1 + g) / (r - g)
    
    # Create the series with historical + projected values
    fcff_series = fcff_historical.copy()
    fcff_series.loc[2026] = fcff_2026
    fcff_series.loc["Terminal Value"] = terminal_value
    
    return fcff_series
##### LLM END #####
