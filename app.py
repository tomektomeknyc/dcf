# app.py
import time
import streamlit as st
from news_fetcher import get_headlines
from stock_analysis_renderer import render_financial_analysis

# ─── 1) Streamlit page config ─────────────────────────────────
st.set_page_config(
    page_title="🚀 Starship Finance Simulator", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)
selected_stocks = st.session_state.get("selected_stocks", [])

# Now safe to use st.session_state and other Streamlit commands
# Show general market headlines
if not selected_stocks or st.checkbox("📰 Show market headlines"):
    st.markdown("### 📰 Market News")
    headlines = get_headlines()

if headlines:
    with st.container():
        # No leading newline in triple quote below!
        content = """<div style="height: 300px; overflow-y: auto; padding: 1em; background-color: #111111; border: 1px solid #444; border-radius: 8px;">"""
        for item in headlines:
            title = item.get("title", "No Title")
            url = item.get("link", "#")
            published = item.get("published", "").replace("T", " ").replace("Z", "")

            content += f"""
<div style="margin-bottom: 1em;">
  <a href="{url}" target="_blank" style="color:#1E90FF; font-weight:600; font-size:16px; text-decoration:none;">{title}</a><br>
  <span style="font-style: italic; color: #BBBBBB;"><i>Published: {published}</i></span>
</div>
"""
        content += "</div>"
        st.markdown(content, unsafe_allow_html=True)
else:
    st.info("No news available right now.")








# Continue the rest of your app only when stocks are selected
if selected_stocks:
    st.markdown(f"### Selected Stocks: {', '.join(selected_stocks)}")
    # Insert your stock analysis rendering functions here
    render_financial_analysis(selected_stocks)



# ─── Initialize our "have we run X?" flags ──────────────────────────────────
for _key in ("ff5_ran", "capm_ran", "damo_ran"):
    if _key not in st.session_state:
        st.session_state[_key] = False
# ─── Initialize our "have we run intrinsic valuation?" flag ─────────────────
if "intrinsic_computed" not in st.session_state:
    st.session_state["intrinsic_computed"] = False


import pandas as pd
# Ensure we have a place to store the Damodaran industry‐beta DataFrame
if "damo_industry_df" not in st.session_state:
    st.session_state["damo_industry_df"] = pd.DataFrame()

from fetch_damodaran_betas import (
    fetch_damodaran_industry_betas,
    find_industry_beta,
    calculate_wacc_with_industry_beta,
    _get_beta_urls,
    _download_region_beta,
    map_folder_to_region, 
)
from project_description_tab import render_project_description_tab
import plotly.express as px
from pathlib import Path
from scrape_ff5 import get_ff5_data_by_folder
import os
import plotly.graph_objects as go
import uuid
from stock_returns import fetch_daily_returns  # to read saved stock‐return CSVs
import numpy as np 
import warnings
from regression_engine import (
    compute_capm_beta,
    compute_ff5_betas,
    compute_ff5_residuals,
    compute_industry_residuals  
)
from ticker_to_industry import ticker_industry_map
from metrics import compute_driver_metrics
from fcf_calculations import compute_fcff 
from dcf_valuation import calculate_all_intrinsic_values
from project_description_tab import render_project_description_tab
from qa_tab import render_qa_tab
from fin_report_tab import render_fin_report_tab

# Show a logo at the top of the sidebar
logo_path = Path(__file__).parent / "attached_assets" / "logo.png"  
if logo_path.exists():
   st.sidebar.image(str(logo_path), width=200, caption="")
# ─── Ticker_to_region ──────────────────────────────────────────────────
def ticker_to_region(ticker: str) -> str:
    parts = ticker.split(".")
    if len(parts) == 1:
        return "US"
    suffix = parts[-1].upper()
    region_map = {
        "AX": "AU_NZ",
        "NZ": "AU_NZ",
        "TO": "AU_NZ",
        "DE": "Europe",
        "O":  "US",
    }
    return region_map.get(suffix, "US")

st.session_state.pop("ff5", None)
# Silence only the "'M' is deprecated" FutureWarning from pandas
warnings.filterwarnings(
    "ignore",
    message=r".*'M' is deprecated and will be removed in a future version.*",
    category=FutureWarning,
)

# ─── 2) Inject external CSS ────────────────────────────────────
with open("styles.css") as f:
    css = f.read()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

if "market_returns" not in st.session_state:
    region_indices = {
        "US": ".SPX",     # S&P 500
        "AU": ".AXJO",  # ASX 200 TR
        "NZ": ".NZ50",    # NZX 50 TR
        "DE": ".GDAXI",   # DAX
    }
    mkt_returns: dict[str, pd.Series] = {}
    for region, ric in region_indices.items():
        fname = ric.lstrip(".").replace(".", "_") + "_market_returns.csv"
        path  = os.path.join("attached_assets", fname)
        if os.path.exists(path):
            df_mkt = pd.read_csv(path, parse_dates=True, index_col=0)
            mkt_returns[region] = df_mkt["MarketReturn"]
        else:
            st.sidebar.warning(f"⚠️ Missing market‐returns file for {region}: {path}")
    st.session_state["market_returns"] = mkt_returns

##### CAPM START #####

def compute_pure_capm_beta(stock_ret: pd.Series, market_ret: pd.Series) -> float:
    """
    Compute β = Cov(Stock, Market) / Var(Market)
    on their overlapping dates.
    """
    df_combined = pd.concat(
        [stock_ret.rename("stock"), market_ret.rename("market")],
        axis=1,
    ).dropna()
    cov_sm = df_combined["stock"].cov(df_combined["market"])
    var_m  = df_combined["market"].var()
    return cov_sm / var_m if var_m else 0.0

if "last_year_wacc" not in st.session_state:
    st.session_state["last_year_wacc"] = None


# ─── 1) Existing loader/grabber ────────────────────────────────────────────
YEAR_ROW = 10
COLS     = list(range(1,16))

def load_sheet(xlsx: Path, sheet: str):
    try:
        df = pd.read_excel(xlsx, sheet_name=sheet, header=None, engine="openpyxl")
    except:
        return None, None
    if df.shape[0] <= YEAR_ROW or df.shape[1] <= max(COLS):
        return None, None
    years = df.iloc[YEAR_ROW, COLS].astype(int).tolist()
    return df, years

def grab_series(xlsx: Path, sheet: str, regex: str):
    df, years = load_sheet(xlsx, sheet)
    if df is None:
        return None
    col0 = df.iloc[:,0].astype(str).str.lower()
    mask = col0.str.contains(regex, regex=True, na=False)
    if not mask.any():
        return None
    row = df.loc[mask, :].iloc[0]
    return pd.to_numeric(row.iloc[COLS], errors="coerce").tolist()
##### CAPM END #####
@st.cache_data
def build_dataset():
    base = Path(__file__).parent
    rows = []

    for xlsx in base.rglob("*.xlsx"):
        ticker = xlsx.stem
        # Record which folder (region) this ticker came from
        region = xlsx.parent.name
        st.session_state.setdefault("region_map", {})[ticker] = region
        # Record the industry for this ticker
        industry = ticker_industry_map.get(ticker, "General")
        st.session_state.setdefault("industry_map", {})[ticker] = industry

        # # Debug: Show what's actually in the Excel file around row 162
        # df_temp, _ = load_sheet(xlsx, "Income Statement")
        # if df_temp is not None:
        #     print(f"DEBUG: Rows 160-165 in column A for {ticker}:")
        #     for i in range(160, 166):
        #         if i < df_temp.shape[0]:
        #             print(f"  Row {i+1}: {df_temp.iloc[i, 0]}")

        # Get years
        _, years = load_sheet(xlsx, "Income Statement")
       

        if years is None:
           continue

        pretax = grab_series(xlsx, "Income Statement", r"income (?:before|pre)[ -]tax")

        taxcash = grab_series(xlsx, "Cash Flow", r"income taxes.*paid")

        # Compute effective tax rate per year (avoid divide‐by‐zero)
        if pretax and taxcash:
            tax_rate_series = [
                (t / p) if p not in (0, None) else 0.0
                for p, t in zip(pretax, taxcash)
            ]
        else:
          # Fallback to zero‐rate if either series is missing
            tax_rate_series = [0.0] * len(years)



        # Grab core series
        ebitda    = grab_series(xlsx, "Income Statement", r"earnings before.*ebitda")
        #ebit      = grab_series(xlsx, "Income Statement", r"ebit")
        ebit = grab_series(xlsx, "Income Statement", r"earnings before interest.*taxes")
        if ebit:
        # print(f"DEBUG: Found EBIT for {ticker}: {ebit[:3]}...")  # Show first 3 values
            capex     = grab_series(xlsx, "Cash Flow",         r"capital expenditure|capex")
            debt      = grab_series(xlsx, "Balance Sheet",     r"total debt|debt\b")
            cash      = grab_series(xlsx, "Balance Sheet",     r"cash and cash equivalents|cash$")
            ev        = grab_series(xlsx, "Financial Summary", r"^enterprise value\s*$")
            taxes_cf  = grab_series(xlsx, "Cash Flow",         r"income taxes\s*-\s*paid")

            dep_amort = grab_series(xlsx, "Income Statement", r"depreciation.*depletion.*amortization.*total")
        #if dep_amort:
           # print(f"DEBUG: Found Depreciation for {ticker}: {dep_amort[:3]}...")

        # Skip if any core series missing
        if None in (ebitda,ebit, capex, debt, cash, ev, taxes_cf):
            continue

        # Compute ΔNWC from Balance Sheet
        curr_assets = grab_series(xlsx, "Balance Sheet", r"total current assets")
        curr_liab   = grab_series(xlsx, "Balance Sheet", r"total current liabilities")

        # ── Grab Sales ─────────────────────────────────────────────────────────
        #----Matches row 21: "Sales of Goods & Services – Net – Unclassified"
##### NWC START #####       
        sales = grab_series(xlsx, "Income Statement", r"sales of goods & services\s*-\s*net")
        shares_outstanding = grab_series(xlsx, "Balance Sheet", r"common shares.*outstanding.*total")

        if curr_assets and curr_liab:
            nwc = [a - l for a, l in zip(curr_assets, curr_liab)]
            change_in_nwc = [0] + [nwc[i] - nwc[i-1] for i in range(1, len(nwc))]
        else:
            change_in_nwc = [0] * len(years)
##### NWC STOP #####
        # Pull interest (IS first, then CF)
        ie_is = grab_series(xlsx, "Income Statement", r"interest expense|finance costs")
        ie_cf = grab_series(xlsx, "Cash Flow",        r"interest\s*paid")
        interest_expense = ie_is if ie_is is not None else (ie_cf or [0] * len(years))

        # Assemble rows
        for i ,(y, e,eb, c, d, ca, v, t, nwc0, ie, da, so, sal) in enumerate(zip(
    years, ebitda,ebit, capex, debt, cash, ev, taxes_cf,
    change_in_nwc, interest_expense, dep_amort, shares_outstanding or [None]*len(years), sales or [None]*len(years)
        )):
            rows.append({
                "Ticker":          ticker,
                "Year":            y,
                "EBITDA":          e,
                "EBIT":            eb,
                "CapEx":           c,
                "Debt":            d,
                "Cash":            ca,
                "EV":              v,
                "CashTaxesPaid":   t,
                "ChangeNWC":       nwc0,
                "InterestExpense": ie,
                "tax_rate":         tax_rate_series[i],
                "DepreciationAndAmortization": da,
                "SharesOutstanding": shares_outstanding[i] if shares_outstanding is not None else None,
                "Sales":           sales[i] if sales is not None else None,
            })

    # Build DataFrame
    df = pd.DataFrame(rows)
    if df.empty:
      return df

    # Historical ΔDebt & ΔCash
    df["ΔDebt"] = df.groupby("Ticker")["Debt"].diff().fillna(0)
    df["ΔCash"] = df.groupby("Ticker")["Cash"].diff().fillna(0)

##### FCFF START #####
# 1) FCFF = EBIT × (1 – tax_rate) + Depreciation & Amortization – CapEx – ΔNWC
    df["FCFF"] = (
    df["EBIT"] * (1 - df["tax_rate"])
  + df["DepreciationAndAmortization"] 
  - df["CapEx"]
  - df["ChangeNWC"]
)
##### FCFF END #####
    # Debug: Print FCFF components for first few rows
    # print("DEBUG: FCFF Components:")
    # for i, row in df.head(3).iterrows():
    # print(f"  {row['Ticker']} ({row['Year']}): EBIT={row['EBIT']}, tax_rate={row['tax_rate']:.3f}, DA={row['DepreciationAndAmortization']}, CapEx={row['CapEx']}, ΔNWC={row['ChangeNWC']}, FCFF={row['FCFF']}")
    
##### FCFE START #####
# 2) FCFE = FCFF – (InterestExpense × (1–tax_rate)) + ΔDebt – ΔCash,
# Using the per‐row tax_rate we built earlier
    df["FCFE"] = (
    df["FCFF"]
  - df["InterestExpense"] * (1 - df["tax_rate"])
  + df["ΔDebt"]
  - df["ΔCash"]
)
##### FCFE END #####
    # ── Free Cash Flow (FCF) including ΔNWC ────────────────────────────
    df["FCF"] = (
    df["EBITDA"]
  - df["CashTaxesPaid"]
  - df["ChangeNWC"]
  - df["CapEx"]
)


# 3) EV/EBITDA
   
    df["EV/EBITDA"] = df["EV"] / df["EBITDA"].replace(0, pd.NA)


##### TV START #####
# ─── 4) Terminal Value Calculation: FCFF₂₆ = FCFF₂₅ × (1 + g) ─────────────────
    terminal_rows = []
    for ticker in df["Ticker"].unique():
        ticker_data = df[df["Ticker"] == ticker].sort_values("Year")
        if not ticker_data.empty:
            # Get the most recent year's FCFF (FCFF₂₅)
            latest_row = ticker_data.iloc[-1]
            fcff_2025 = latest_row["FCFF"]
            
            # Simple growth rate assumption (can be made dynamic later)
            g = 0.03  # 3% perpetual growth rate
            
            # Calculate FCFF₂₆ = FCFF₂₅ × (1 + g)
            fcff_2026 = fcff_2025 * (1 + g)
            
            # Create Terminal Value row
            terminal_row = latest_row.copy()
            terminal_row["Year"] = "Terminal Value"
            terminal_row["FCFF"] = fcff_2026
            terminal_rows.append(terminal_row)
    
    # Append Terminal Value rows to the main DataFrame
    if terminal_rows:
        terminal_df = pd.DataFrame(terminal_rows)
        df = pd.concat([df, terminal_df], ignore_index=True)



##### TV END #####
    # Create empty DataFrames for Cash Flow and Balance Sheet masters
    df_cf_master = pd.DataFrame()  # ✅ FIXED: Now properly defined
    df_bs_master = pd.DataFrame()  # ✅ FIXED: Now properly defined

    return df, df_cf_master, df_bs_master

df, df_cf, df_bs = build_dataset()
if df.empty:
    st.error("❌ No data found. Check your folders/sheets.")
    st.stop()

# Filter out Terminal Value rows for driver metrics calculation (only use historical data)
df_historical = df[df["Year"] != "Terminal Value"]
drivers = compute_driver_metrics(df_historical, lookback=5)
st.session_state["drivers"] = drivers


# ─── 3) Sidebar: selectors & sliders ─────────────────────────────────────────────
tickers     = sorted(df["Ticker"].unique())
sel_tickers = st.sidebar.multiselect("🔎 Companies", options=tickers, default=[])

for ticker in sel_tickers:
    # slice out only the historical rows for that ticker
    df_is = df[df["Ticker"] == ticker].set_index("Year")
    
    # Temporarily skip FCFF calculation until I implement the simple version
    fcff_series = pd.Series(dtype=float, name='FCFF')

    # Now fcff_series has indices ..., 2026, and "Terminal Value"
    #st.write(f"### {ticker} FCFF + Terminal Value")
    #st.dataframe(fcff_series)

if "prev_sel_tickers" not in st.session_state:
    st.session_state.prev_sel_tickers = sel_tickers.copy()

# Figure out which tickers were just removed:
# ─── Remove any state for tickers that were just un‐selected ─────────────────
removed = set(st.session_state.prev_sel_tickers) - set(sel_tickers)
if removed:
    for t in removed:
# Clear FF-5, CAPM & Damodaran dicts
        for key in (
            "stock_returns", "ff5_resids", "capm_resids",
            "capm_results", "damo_resids"
        ):
            if key in st.session_state:
                st.session_state[key].pop(t, None)
# Purge those tickers out of your Damodaran history list
    st.session_state["damo_history"] = [
        entry for entry in st.session_state.get("damo_history", [])
        if entry and entry[0] not in removed
    ]
# Remember the new selection
st.session_state.prev_sel_tickers = sel_tickers.copy()

# Pick a palette and assign each ticker a fixed color:
default_colors = px.colors.qualitative.Plotly
color_map = {
    t: default_colors[i % len(default_colors)]
    for i, t in enumerate(sel_tickers)
}

# ─── 4) Build shared color map ────────────────────────────────────────────────

default_colors = px.colors.qualitative.Plotly
color_map = {
    t: default_colors[i % len(default_colors)]
    for i, t in enumerate(sel_tickers)
}

if not sel_tickers:
    st.sidebar.info("Please select at least one company to continue.")
    st.stop()

years_avail  = df[df.Ticker.isin(sel_tickers)]["Year"].dropna().unique()
years_sorted = sorted(int(y) for y in years_avail if str(y).isdigit())
if not years_sorted:
    st.sidebar.error("No years available for the selected companies.")
    st.stop()

# ─── 1) Single-year selector for all your point-in-time metrics ───────────────
sel_year = st.sidebar.slider(
    "📅 Year",
    min_value=years_sorted[0],
    max_value=years_sorted[-1],
    value=years_sorted[-1],
)

# ─── 2) Alpha/Beta time-window selector (tuple) ───────────────────────────────
alpha_start, alpha_end = st.sidebar.slider(
    "📈 Alpha & Beta Date Range",
    min_value=2015,
    max_value=max(sel_year, 2016),  # Ensure max is at least 2016
    value=(2015, max(sel_year, 2016)),   # Ensure default value is valid
    format="%d",
    key=f"alpha_beta_{sel_year}"
)


st.sidebar.markdown(
    f"ℹ️ Note: some tickers (e.g. AIR.NZ) only have data through 2024. "
    "Move the slider down to see their multiples."
)


# ─── 3a) Choose Estimation Methods ─────────────────────────────────────────────
st.sidebar.markdown(
    """
    <span>⚙️ <strong>Estimation Methods</strong></span>
    <span style='color:gray; cursor:help;' title='When all (FF5, CAPM, Damo α) are selected, the Intrinsic Value Table will appear below.'>ℹ️</span>
    """,
    unsafe_allow_html=True
)



methods = st.sidebar.multiselect(
    label="",
    options=["Historical", "FF-5", "CAPM", "Damo α"],
    default=["Historical"],
)
# As soon as the user picks “Damo α”, mark it run
if "Damo α" in methods:
    st.session_state["damo_ran"] = True

# with st.expander("Show which models have run", expanded=False):
#     st.write({
#         "ff5_run":  st.session_state["ff5_ran"],
#         "capm_run": st.session_state["capm_ran"],
#         "damo_run": st.session_state["damo_ran"]
#     })

###################### Damodaran Betas ########################
##### DAMODARAN START #####
# — Pre-load Damodaran betas for each region, but cache locally —
damo_files: dict[str,str] = {}
for reg in ("US", "Europe", "AU_NZ"):
    local_path = Path("Damodaran") / f"totalbeta{reg}.xls"
    if local_path.exists():
        # if we already downloaded it, just use the local file
        damo_files[reg] = str(local_path)
    else:
        # otherwise download and save
        damo_files[reg] = _download_region_beta(reg)
st.session_state["damo_files"] = damo_files
# Fetch & cache the industry‐beta table for the first selected ticker’s region
if sel_tickers:
    first = sel_tickers[0]
    region0 = ticker_to_region(first)
    st.session_state["damo_industry_df"] = fetch_damodaran_industry_betas(region0)


# ─── Sidebar toggle for Damodaran β in the combined β-chart ─────────────
show_damo = st.sidebar.checkbox(
    "Show Damodaran β in the combined β-chart",
    value=False,
    key="show_damo_chart",

)

# ─── Damodaran Industry Betas (sidebar) ─────────────────────────────────────

if sel_tickers:
    damo_betas = {}
    for t in sel_tickers:
        # — Pull off the suffix (e.g. 'O', 'AX', 'NZ', 'DE', etc.)
        parts  = t.split(".")
        suffix = parts[-1].upper() if len(parts) > 1 else "O"
        # — Now map that suffix into your Damodaran region key
        region = map_folder_to_region(suffix)

        # — Fetch the industry‐beta table for the right region
        damo_df = st.session_state["damo_industry_df"]


# Show me every industry label in that sheet
# (Then delete/comment once you’ve verified your mapping)
#  st.sidebar.write(f"Available industries in {region}:", df_damo["Industry"].unique())
# — Look up the industry name you wired up in ticker_to_industry.py
        industry = ticker_industry_map.get(t)
# – Fetch the industry‐beta table for the right region
        df_damo = st.session_state["damo_industry_df"]

# — Do your find_industry_beta exactly once
        match = find_industry_beta(
            df_damo,
            company_sector   = None,
            company_industry = industry,
        )
        damo_betas[t] = match["beta"] if match else None



# — Render the little green box —
st.sidebar.markdown("### 📟 Damodaran Industry Betas")
beta_slot = st.sidebar.empty()

# Show only the most‐recently loaded β
if damo_betas:
    last_tkr, last_beta = next(reversed(damo_betas.items()))
    if last_beta is not None:
        beta_slot.success(f"{last_tkr} β = {last_beta:.2f}")
    else:
        beta_slot.write(f"{last_tkr} β = n/a")
##### DAMODARAN END #####

# — Trimmed table: only Ticker, Industry, and β —

# Only show the sidebar table if you have tickers selected AND at least one beta type is enabled
any_beta = any(m in methods for m in ["FF-5", "CAPM", "Damo α"])
if sel_tickers and any_beta:
    # — Trimmed table: only Ticker, Industry, and β —
    sidebar_df = pd.DataFrame([
        {
            "Ticker":        t,
            "Industry":      ticker_industry_map.get(t, "n/a"),
            "β":             damo_betas.get(t, None),
        }
        for t in sel_tickers
    ]).set_index("Ticker")

    st.sidebar.dataframe(
        sidebar_df,
        height=300,
        use_container_width=True,
    )
# — Create two tabs: Main vs Project Description —
tabs = st.tabs(["Main", "Project Description", "Q&A", "Fin Report Generator"])
tab_main, tab_desc, tab_qa, tab_fin_report = tabs

with tab_main:
    with st.expander("Show which models have run", expanded=False):
        st.write({
        "ff5_run": st.session_state["ff5_ran"],
        "capm_run": st.session_state["capm_ran"],
        "damo_run": st.session_state["damo_ran"],
    })

    damo_history = st.session_state.get("damo_history", [])


    # ─── Damodaran Industry Betas ────────────────────────────────

    # Pick the *first* selected ticker 
    t = sel_tickers[0]
    region = ticker_to_region(t)
    industry = ticker_industry_map.get(t)


    # Fetch the right region‐sheet and store in session_state
    st.session_state["damo_industry_df"] = fetch_damodaran_industry_betas(region)
    damo_df = st.session_state["damo_industry_df"]



    if industry is None:
        st.sidebar.error(f"❌ No industry mapping for {t}")
    elif damo_df.empty:
        st.sidebar.error("❌ Could not load Damodaran betas")
    else:
    # Find_industry_beta will look up the exact row
        match = find_industry_beta(
            industry_betas_df = damo_df,
            company_sector    = region,
            company_industry  = industry
        )
        if not match:
            st.sidebar.error(f"❌ No β found for industry '{industry}' in {region}")
        else:
    # Show the full table, *and* highlight the matched row:
    # Note: This duplicate section has been removed to fix the 0.00 beta display issue
            pass



    def ticker_to_region(ticker: str) -> str:
        parts = ticker.split(".")
        if len(parts) == 1:
    # No exchange suffix → default to US
            return "US"

        suffix = parts[-1].upper()
    # Map suffixes to your folder names
        region_map = {
            "AX": "AU",   # Australian tickers
            "NZ": "NZ",   # New Zealand tickers
            "DE": "DE",   # German tickers
            "O":  "US",   # .O suffix → US
        }
        return region_map.get(suffix, "US")


##### FAMA FRENCH START #####

    # ─── FF-5: download, compute alphas & cache ─────────────────────────────
    if "FF-5" in methods:
        st.session_state["ff5_ran"] = True
        ph_ff5 = st.sidebar.empty()
        ph_ff5.markdown("#### 🔢 Auto‑Updating FF‑5 Factors")
        
        
       
        # Auto-fetch FF-5 data when method is selected
        ff5_results: dict[str, pd.DataFrame] = {}
        for ticker in sel_tickers:
            folder = ticker_to_region(ticker)
            try:
                df_ff5 = get_ff5_data_by_folder(ticker, folder)
                ff5_results[ticker] = df_ff5
            except Exception as e:
                st.sidebar.error(f"❌ {ticker}: {e}")

        if ff5_results:
            st.session_state["ff5"] = ff5_results
            st.session_state["ff5_ran"] = True
            time.sleep(5)
            ph_ff5.empty()
    # 2) Compute & cache pure-alpha residuals
            ff5_df     = st.session_state["ff5"]                # dict[ticker, DataFrame]

##### FAMA FRENCH END #####

    # ── 1a) Incrementally fetch only *new* stock returns ────────────────────

            stock_rets = st.session_state.get("stock_returns", {})
            for ticker in sel_tickers:
                if ticker in stock_rets:
                    continue
                fn = f"attached_assets/returns_{ticker.replace('.', '_')}.csv"
                if os.path.isfile(fn):
                    df_ret = pd.read_csv(fn, parse_dates=True, index_col=0)
                    stock_rets[ticker] = df_ret["Return"]
                else:
                    st.sidebar.error(f"Missing returns file for {ticker}: {fn}")
            st.session_state["stock_returns"] = stock_rets


            ff5_resids = {
                t: compute_ff5_residuals(
                    stock_rets[t],  # pd.Series of daily returns for ticker t
                    ff5_df[t].apply(pd.to_numeric, errors="coerce") 
                )
                for t in sel_tickers
                if t in stock_rets and t in ff5_df
            }
            st.session_state["ff5_resids"] = ff5_resids



    # ─── Later, if you need to recompute in a loop ────────────────────────────
    if "ff5" in st.session_state:
        ff5_dict   = st.session_state["ff5"]                # dict[ticker, DataFrame]
        stock_rets = st.session_state.get("stock_returns", {})  # dict[ticker, Series]

        ff5_resids: dict[str, pd.Series] = {}
        for ticker, daily_ret in stock_rets.items():
            if ticker not in ff5_dict:
                continue

    # Roll daily → monthly and name the series "y"
            monthly = (1 + daily_ret).resample("M").prod() - 1
            monthly = monthly.rename("y")
            if hasattr(monthly.index, 'to_period'):
                monthly.index = monthly.index.to_period("M").to_timestamp()

    # Call compute_ff5_residuals with the Series then that ticker's factor DataFrame
            ff5_resids[ticker] = compute_ff5_residuals(
                monthly,          # pd.Series named "y"
                ff5_dict[ticker].apply(pd.to_numeric, errors="coerce")
            )

        st.session_state["ff5_resids"] = ff5_resids

    # Recompute betas & errors only when the ticker set changes
        if set(ff5_dict) != set(st.session_state.get("ff5_betas", {})):
                # 1) Annualize RF & market premium
                sample_ff5     = next(iter(ff5_dict.values()))
                sample_ff5     = sample_ff5.apply(lambda col: pd.to_numeric(col, errors="coerce"))
                monthly_rf     = sample_ff5["RF"].mean()    / 100
                monthly_mktrf  = sample_ff5["Mkt-RF"].mean() / 100

                rf_annual      = (1 + monthly_rf)   ** 12 - 1
                mktprem_annual = (1 + monthly_mktrf) ** 12 - 1

##### BETAS START #####
    # 2) Compute fresh betas & regression errors
                betas_by_ticker  = {}
                errors_by_ticker = {}
                for ticker, ff5_df in ff5_dict.items():
                    path = f"attached_assets/returns_{ticker.replace('.', '_')}.csv"
                    if not os.path.isfile(path):
                        st.sidebar.error(f"Missing returns file for {ticker}: {path}")
                        continue
                    returns_df = pd.read_csv(path, parse_dates=True, index_col=0)
                    stock_ret  = returns_df["Return"]
                    res = compute_ff5_betas(stock_ret, ff5_df)
                    betas_by_ticker[ticker] = {
                        "Mkt-RF": res["market_beta"],
                        "SMB":    res["smb_beta"],
                        "HML":    res["hml_beta"],
                        "RMW":    res["rmw_beta"],
                        "CMA":    res["cma_beta"],
                    }
                    errors_by_ticker[ticker] = {
                        "r_squared": res["r_squared"],
                        "alpha":     res["alpha"],
                    }

    # 3) Store in session
                st.session_state["ff5_betas"]  = betas_by_ticker
                st.session_state["ff5_errors"] = errors_by_ticker

##### BETAS END #####

    # Only flash “recalculating” if slider‐year really changed
                last_year   = st.session_state.get("last_year_wacc")
                do_message  = (last_year is not None and last_year != sel_year)

    # ——————————————————————————————————————————————————————————————————————————————
    # 3 Pure CAPM regression + FF-5 residuals time series
    # ——————————————————————————————————————————————————————————————————————————————

##### CAPM START #####

    if "CAPM" in methods:
        ph = st.sidebar.empty() #<-- new
        ph.markdown("#### 🧮 Auto‑Updating Pure‑CAPM Regression")
          # wait for 5 seconds
        time.sleep(5)

        # clear the markdown message
        ph.empty() 

        # Auto-compute CAPM when method is selected
        # 1) Prepare containers
        capm_results: dict[str, dict[str, float]] = {}
        stock_rets: dict[str, pd.Series]    = {}
        capm_resids:  dict[str, pd.Series]    = {}    # ← initialize CAPM residuals


    # 2) Loop tickers to load returns & compute β
        for ticker in sel_tickers:
            with st.spinner(f"Computing pure-CAPM β for {ticker}…"):
    # — Load  CSV of daily returns
                csv_p = f"attached_assets/returns_{ticker.replace('.', '_')}.csv"
                try:
                    df_ret = pd.read_csv(csv_p, parse_dates=True, index_col=0)
                except FileNotFoundError:
                    st.sidebar.error(f"❌ No returns CSV for {ticker}, run fetch first")
                    continue

    # Extract the daily return series
                stock_ret = df_ret["Return"].dropna()
    # Cache it
                stock_rets[ticker] = stock_ret

    # choomonthly = (1 + stock_ret).resample("M").prod() - 1se the right market series
                region     = ticker_to_region(ticker)
                market_ret = st.session_state["market_returns"].get(region)
                if market_ret is None:
                    st.sidebar.error(f"❌ No market returns for region {region}")
                    continue

    # Compute the CAPM β
                try:
                    beta = compute_pure_capm_beta(stock_ret, market_ret)
                    capm_results[ticker] = {"beta": beta}
                    #st.write("DEBUG capm_results for", ticker, "→", capm_results[ticker])

                    # ── Compute CAPM residuals (α) ──────────────────────────
                    # Monthly stock returns
                    monthly = (1 + stock_ret).resample("M").prod() - 1
                    if hasattr(monthly.index, 'to_period'):
                        monthly.index = monthly.index.to_period("M").to_timestamp()
                    # Monthly market returns
                    mkt_month = (1 + market_ret).resample("M").prod() - 1
                    if hasattr(mkt_month.index, 'to_period'):
                        mkt_month.index = mkt_month.index.to_period("M").to_timestamp()
                    # Residual = actual – β×market
                    resid = monthly - beta * mkt_month
                    capm_resids[ticker] = resid

##### CAPM END #####

                except Exception as e:
                    st.sidebar.error(f"❌ {ticker}: {e}")

        # 3) Cache into session_state
            st.session_state["stock_returns"] = stock_rets
            st.session_state["capm_results"]  = capm_results
            st.session_state["capm_ran"]      = True
            if st.session_state.get("capm_ran") and not st.session_state.get("capm_notified", False):
                st.sidebar.success("✅ Pure-CAPM betas & returns cached")
                st.session_state["capm_notified"] = True
                st.session_state["capm_resids"]   = capm_resids   # ← now store the residuals
                # Clear the “Auto-Updating Pure–CAPM Regression” banner
                time.sleep(5)
                ph.empty()
        # ─── NOW grab the FF-5 factors behind your risk-free & market prem
            if "ff5" not in st.session_state:
                ff5_results = {}
                for ticker in sel_tickers:
                    folder = ticker_to_region(ticker)
                    with st.spinner(f"Downloading FF-5 for {ticker}…"):
                        ff5_results[ticker] = get_ff5_data_by_folder(ticker, folder)
                st.session_state["ff5"] = ff5_results

    # Compute annualized RF & market premium exactly once
            sample = next(iter(st.session_state["ff5"].values()))
            rf_monthly     = pd.to_numeric(sample["RF"],    errors="coerce").mean()    / 100
            mktprem_monthly = pd.to_numeric(sample["Mkt-RF"], errors="coerce").mean() / 100

            st.session_state["rf_annual"]     = (1 + rf_monthly)    ** 12 - 1
            st.session_state["mktprem_annual"] = (1 + mktprem_monthly) ** 12 - 1


    # Now compute & cache FF-5 residuals (pure α)
            ff5_resids: dict[str, pd.Series] = {}
            ff5_data   = st.session_state.get("ff5", {})
            for ticker, stock_ret in stock_rets.items():
                if ticker not in ff5_data:
                    continue
                # 1) Roll daily → monthly & align to FF-5 dates
                monthly = (1 + stock_ret).resample("M").prod() - 1
                if hasattr(monthly.index, 'to_period'):
                    monthly.index = monthly.index.to_period("M").to_timestamp()

                # 2) Pull & coerce the FF-5 factors
                ff5_df = ff5_data[ticker].apply(pd.to_numeric, errors="coerce")

                # 3) Re-index your returns to the same dates & force float dtype
                common = monthly.reindex(ff5_df.index).astype(float)


                # Compute residuals
                alpha_ts = compute_ff5_residuals(common, ff5_df)
                ff5_resids[ticker] = alpha_ts

                st.session_state["ff5_resids"] = ff5_resids


    st.sidebar.markdown("### 🎛 Simulations")
    ebt_adj  = st.sidebar.slider("EBITDA Δ%", -50, 50, 0)
    cpx_adj  = st.sidebar.slider("CapEx Δ%",  -50, 50, 0)
    debt_adj = st.sidebar.slider("Debt Δ%",   -50, 50, 0)
    cash_adj = st.sidebar.slider("Cash Δ%",   -50, 50, 0)
    nwc_adj  = st.sidebar.slider("NWC Δ%", -50, 50, 0)

    # ─── Compute the exact historical EV/EBITDA for this year
    base = df.query("Year == @sel_year and Ticker in @sel_tickers").copy()
    # 1) Historical EV and EBITDA
    hist_ev     = base["EV"].sum(skipna=True)
    hist_ebit   = base["EBITDA"].sum(skipna=True)

    # 2) Historical net debt = Debt − Cash
    hist_net_debt = (
        base["Debt"].sum(skipna=True)
    - base["Cash"].sum(skipna=True)
    )

    # 3) Unlevered EV = EV_hist − NetDebt_hist
    unlev_ev    = hist_ev - hist_net_debt

    # 4) Unlevered multiple
    unlev_mult  = (unlev_ev / hist_ebit) if hist_ebit else 0.0


    ev_mult_full = (hist_ev / hist_ebit) if hist_ebit else 0

    # ─── EV/EBITDA slider with two‐decimal steps and a dynamic key
    ev_mult_full = (hist_ev / hist_ebit) if hist_ebit else 0

    ev_mult = st.sidebar.slider(
        "EV/EBITDA (unlevered)",
        min_value=0.10,
        max_value=100.00,
        value=round(unlev_mult, 2),   # default to unlevered multiple
        step=0.01,
        key=f"ev_mult_{sel_year}"
    )


    # ─── 4) Filter & simulate ───────────────────────────────────────────────────────
    base = df.query("Year == @sel_year and Ticker in @sel_tickers").copy()
    if base.empty:
        st.warning("No data for that selection.")
        st.stop()

    sim = base.copy()
    # Adjust everything except Cash multiplicatively
    for col, pct in [
        ("EBITDA", ebt_adj),
        ("CapEx",  cpx_adj),
        ("Debt",   debt_adj),
    ]:
        sim[col] = sim[col] * (1 + pct / 100)

    # Adjust Cash so +% always moves the balance toward +∞
    sim["Cash"] = base["Cash"] + base["Cash"].abs() * (cash_adj / 100)


    # Apply the NWC % slider BEFORE OCF
    sim["ChangeNWC"]      = base["ChangeNWC"] * (1 + nwc_adj / 100)

    with st.expander("ChangeNWC", expanded=False):
        st.write(sim["ChangeNWC"].tolist())



    # Keep the historical cash taxes constant unless you add a slider for it
    sim["CashTaxesPaid"]  = base["CashTaxesPaid"]

    # 1) Recalc OCF = EBITDA – CashTaxesPaid – ΔNWC (ΔNWC still zero in this simple sim)
    sim["OCF"]   =    (      sim["EBITDA"] - sim["CashTaxesPaid"]- sim["ChangeNWC"])

    # 2) FCF = OCF – CapEx
    sim["FCF"]            = sim["OCF"] - sim["CapEx"]
    #st.write("🔍 sim FCF after NWC adj:", sim["FCF"].tolist())
    with st.expander("🔍 sim FCF after NWC adj:", expanded=False):
        st.write(sim["FCF"].tolist())
##### EV/EBITDA START #####
    # 3) EV and EV/EBITDA 

    # Recompute sim net‐debt
    sim_net_debt = sim["Debt"] - sim["Cash"]

    # EV = EBITDA × unlevered‐multiple + change in net debt
    sim["EV"] = sim["EBITDA"] * ev_mult + (sim_net_debt - hist_net_debt)

    # sim["EV"] = sim["EBITDA"] * ev_mult

    sim["EV/EBITDA"]      = sim["EV"] / sim["EBITDA"].replace(0, pd.NA)



    # ─── 5) Top metrics: two‐row panels, 5 columns each ────────────────────────────
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
        ("Tax Rate",  "tax_rate",    "{:.1%}"), 
    ]
##### EV/EBITDA END #####

    # First 5 always go on row 1, next 4 on row 2 (with one blank placeholder)
    #Two rows of 5 metrics each
    first5 = hist_metrics[:5]
    rest5  = hist_metrics[5:10]    # exactly the other five

    # ─── Create scrollable container for stock metrics ─────────────────────────────
    st.markdown("## 🏦 Company Metrics Overview")
    # Create a scrollable container using Streamlit's container with CSS height control
    with st.container(height=600):
        for ticker in sel_tickers:
            t_base = base[base.Ticker == ticker]
            if t_base.empty:
                continue
            t_sim = sim[sim.Ticker == ticker]

            st.markdown(
            f"""
            <span style="font-family: Orbitron, monospace; color: #39FF14; font-size: 1.8em; letter-spacing: 2px;">
                {ticker} – Year {sel_year}
            </span>
            """,
            unsafe_allow_html=True,
        )
#################################################################

        # — Implied‐multiple cross‐check ——————————————————————————
    
            # ebitda    = t_base["EBITDA"].sum(skipna=True)
            # ev        = t_base["EV"].sum(skipna=True)
            # ev_ebitda = ev / ebitda if ebitda else pd.NA
            # debt      = t_base["Debt"].sum(skipna=True)
            # cash      = t_base["Cash"].sum(skipna=True)
            # #st.write("▶️ t_base columns:", list(t_base.columns))
            # shares    = t_base["SharesOutstanding"].sum(skipna=True)

            # # calculate implied values
            # implied_ev     = ebitda * ev_ebitda
            # implied_equity = implied_ev - (debt - cash)
            # implied_price  = implied_equity / shares if shares else pd.NA

            # # render three check‐cards
            # cols = st.columns(3)
            # cols[0].metric("Implied EV",          f"${implied_ev:,.0f}")
            # cols[1].metric("Implied Equity",      f"${implied_equity:,.0f}")
            # cols[2].metric("Implied Price/Share", f"${implied_price:,.2f}")
    
################################################################

            # ─ Historical Metrics ─────────────────────────────────────────────────────
            st.markdown("#### Historical Metrics")
            # Row 1: first 5
            cols = st.columns(5)
            for (label, field, fmt), col in zip(first5, cols):
                if field == "EV/EBITDA":
                    total_ebit = t_base["EBITDA"].sum(skipna=True)
                    val = (t_base["EV"].sum(skipna=True) / total_ebit) if total_ebit else pd.NA
                else:
                    val = t_base[field].sum(skipna=True)
                col.metric(label, fmt.format(val) if pd.notna(val) else "n/a")

            # Row 2: next 4 + blank
            cols = st.columns(5)
            for (label, field, fmt), col in zip(rest5, cols):
                if label is None:
                    col.write("")  # placeholder
                else:
                    val = t_base[field].sum(skipna=True)
                    col.metric(label, fmt.format(val) if pd.notna(val) else "n/a")

            # ─ Simulated Metrics ─────────────────────────────────────────────────────
            st.markdown("#### Simulated Metrics")
            # Row 1: first 5
            cols = st.columns(5)
            for (label, field, fmt), col in zip(first5, cols):
                if field == "EV/EBITDA":
                    total_ebit = t_base["EBITDA"].sum(skipna=True)
                    hist_val   = (t_base["EV"].sum(skipna=True) / total_ebit) if total_ebit else pd.NA
                    sim_val    = t_sim["EV/EBITDA"].iat[0]
                else:
                    hist_val = t_base[field].sum(skipna=True)
                    sim_val  = t_sim[field].iat[0]
                delta = ""
                if pd.notna(hist_val) and pd.notna(sim_val) and hist_val:
                    delta = f"{sim_val/hist_val - 1:+.1%}"
                col.metric(label, fmt.format(sim_val) if pd.notna(sim_val) else "n/a", delta,
                        help="FCF = EBITDA − CapEx" if field=="FCF" else "")

            # Row 2: next 4 + blank
            cols = st.columns(5)
            for (label, field, fmt), col in zip(rest5, cols):
                if label is None:
                    col.write("")  # placeholder
                else:
                    hist_val = t_base[field].sum(skipna=True)
                    sim_val  = t_sim[field].iat[0]
                    delta = ""
                    if pd.notna(hist_val) and pd.notna(sim_val) and hist_val:
                        delta = f"{sim_val/hist_val - 1:+.1%}"
                    col.metric(label, fmt.format(sim_val) if pd.notna(sim_val) else "n/a", delta,
                            help="FCF = EBITDA − CapEx" if field=="FCF" else "")
            
            # Add separator between companies
            st.markdown("---")

        # Close scrollable container
        st.markdown("</div>", unsafe_allow_html=True)

    # ─── 6) 3D Simulation: EBITDA vs CapEx vs EV ───────────────────────────────────
    st.markdown("### 🔭 3D Simulation: EBITDA vs CapEx vs EV")

    # 1) Build combined DataFrame (must come after sim["FCF"] is recalculated)
    plot_df = pd.concat([
        base.assign(Type="Base"),
        sim.assign(Type="Simulated"),
    ])
    plot_df["FCF_mag"]   = plot_df["FCF"].abs().fillna(0)
    plot_df["FCF_label"] = plot_df["FCF"].apply(lambda x: "Positive" if x >= 0 else "Negative")

    # 2) Compute raw min/max for each axis
    eb_min, eb_max = plot_df["EBITDA"].min(), plot_df["EBITDA"].max()
    cx_min, cx_max = plot_df["CapEx"].min(),  plot_df["CapEx"].max()
    ev_min, ev_max = plot_df["EV"].min(),     plot_df["EV"].max()

    # 3) Add padding so the cube isn’t cramped
    pad = 0.05
    def padded(rmin, rmax, pad):
        span = rmax - rmin
        if span == 0:
            buffer = abs(rmin) * pad if rmin != 0 else 1
            return [rmin - buffer, rmax + buffer]
        return [rmin - pad * span, rmax + pad * span]
    x_range = padded(eb_min, eb_max, pad)
    y_range = padded(cx_min, cx_max, pad)
    z_range = padded(ev_min, ev_max, 0.001)

    # 4) Draw the 3D scatter, sizing by the updated FCF_mag
    fig3d = px.scatter_3d(
        plot_df,
        x="EBITDA", y="CapEx", z="EV",
        color="FCF_label",
        color_discrete_map={"Negative":"red","Positive":"green"},
        symbol="Type",
        size="FCF_mag",
        size_max=26,                # ↑ bigger max so changes are obvious
        hover_name="Ticker",
        hover_data={
            "Type":      True,
            "EBITDA":    ":.2f",
            "CapEx":     ":.2f",
            "EV":        ":.2f",
            "Debt":      ":.2f",
            "Cash":      ":.2f",
            "EV/EBITDA": ":.2f",
            "FCF":       ":.2f",
            "FCF_mag":   ":.4f",      # show magnitude to 4 decimals
        },
        template="plotly_dark",
        title=f"Year {sel_year}: Base vs Simulated"
    )

    # 5) Add the cube wireframe
    import plotly.graph_objects as go
    x0, x1 = x_range; y0, y1 = y_range; z0, z1 = z_range
    cube_x = [x0,x1,None, x1,x1,None, x1,x0,None, x0,x0,  x0,x1,None, x1,x1,None, x1,x0,None, x0,x0,  x0,x0,None, x1,x1,None, x1,x1,None, x0,x0]
    cube_y = [y0,y0,None, y0,y1,None, y1,y1,None, y1,y0,  y0,y0,None, y0,y1,None, y1,y1,None, y1,y0,  y0,y0,None, y1,y1,None, y1,y1,None, y0,y0]
    cube_z = [z0,z0,None, z0,z0,None, z0,z0,None, z0,z0,  z1,z1,None, z1,z1,None, z1,z1,None, z1,z1,  z0,z1,None, z0,z1,None, z0,z1,None, z0,z1]
    fig3d.add_trace(go.Scatter3d(
        x=cube_x, y=cube_y, z=cube_z,
        mode='lines',
        line=dict(color="rgba(200,200,200,0.3)", width=1),
        showlegend=False
    ))

    # 6) Lock axis ranges + view
    fig3d.update_layout(
        margin=dict(l=0,r=0,t=40,b=0),
        width=800, height=600,
        uirevision="fixed_view",
        scene=dict(
            aspectmode="cube",
            xaxis=dict(autorange=False, range=x_range, showbackground=True, backgroundcolor="rgba(20,20,20,0.5)"),
            yaxis=dict(autorange=False, range=y_range, showbackground=True, backgroundcolor="rgba(20,20,20,0.5)"),
            zaxis=dict(autorange=False, range=z_range, showbackground=True, backgroundcolor="rgba(20,20,20,0.5)"),
            camera=dict(eye=dict(x=1.8,y=1.4,z=1.2))
        )
    )

    # 7) Render it
    st.plotly_chart(fig3d, use_container_width=True, key="ev_cube_chart")


    # ─── 7) EV/EBITDA & FCF Over Time ───────────────────────────────────────────────
    st.markdown("### 🔄 EV/EBITDA & FCF Over Time")
    time_df = df[df.Ticker.isin(sel_tickers)].copy()
    time_df["FCF"] = time_df["EBITDA"] - time_df["CapEx"]
    fig2 = px.line(
        time_df, x="Year", y=["FCF","EV/EBITDA"],
        color="Ticker", color_discrete_map=color_map, markers=True,
        template="plotly_dark",
        labels={
            "value":"FCF (USD) / EV/EBITDA (x)",
            "variable":"Metric",
            "Ticker":"Company",
        },
    )
    st.plotly_chart(fig2, use_container_width=True, key="ev_cube_chart2")

    # ─── 8) Data Table ─────────────────────────────────────────────────────────────
    st.markdown("### 📊 Data Table")
    st.dataframe(
        sim[[
            "Ticker","Year","EBITDA","CapEx",
            "FCF","FCFF","FCFE",
            "ChangeNWC","InterestExpense",
            "EV","EV/EBITDA","Debt","Cash"
        ]],
        use_container_width=True, height=300
    )
    # ─── 9) Prepare β‐series for FF-5 and Pure-CAPM ────────────────────────────────

    # 9-0) Grab everything from session (empty dict if missing)
    all_ff5_betas = st.session_state.get("ff5_betas", {})
    all_capm      = st.session_state.get("capm_results", {})
    capm_ran = st.session_state.get("capm_ran", False)
    ff5_ran  = st.session_state.get("ff5_ran", False)


    # 9-1) Filter to only the tickers the user actually selected
    betas = {
        t: all_ff5_betas[t]
        for t in sel_tickers
        if t in all_ff5_betas
    }

    capm = {
        t: all_capm[t]
        for t in sel_tickers
        if t in all_capm
    }

    # (Optional) show FF-5 regression errors
    errors = {
        t: st.session_state.get("ff5_errors", {}).get(t)
        for t in sel_tickers
        if t in st.session_state.get("ff5_errors", {})
    }
    if errors:
        st.markdown("#### FF-5 Regression Errors")
        df_err = pd.DataFrame.from_dict(errors, orient="index")
        st.dataframe(df_err.style.format({
            "r_squared": "{:.2f}",
            "alpha":     "{:.4f}",
        }))


    # 🔍 Scrollable FF-5 α preview
    ff5_resids = st.session_state.get("ff5_resids", {})   # dict[ticker, Series]

    # 1) Turn into a DataFrame, coerce the index once
    df_alpha = pd.DataFrame(ff5_resids)
    df_alpha.index = pd.to_datetime(df_alpha.index, errors="coerce")
    df_alpha.index.name = "Date"

    # 2) Drop columns for deselected tickers
    keep = [t for t in sel_tickers if t in df_alpha.columns]
    df_alpha = df_alpha[keep]

    # 3) Trim to your Alpha/Beta date window
    df_alpha = df_alpha.loc[
        f"{alpha_start}-01-01" : f"{alpha_end}-12-31"
    ]

    # 4) Format index and display
    df_alpha.index = df_alpha.index.strftime("%Y-%m-%d")
    if "FF-5" in methods and st.session_state.get("ff5_ran", False):
        st.markdown("🔍 FF-5 Alphas Over Time (scrollable)")
        st.dataframe(df_alpha, height=400)

    # ─── 9-2) Build Damodaran β’s for each selected ticker ────────────────────────


    damo_betas: dict[str, float] = {}
    for t in sel_tickers:
        # figure out region (US, Europe, AU_NZ, etc.)
        folder_code   = ticker_to_region(t)
        damo_region   = map_folder_to_region(folder_code)

        damo_df = st.session_state["damo_industry_df"]


        industry_name = ticker_industry_map.get(t)
        if not industry_name or damo_df.empty:
            damo_betas[t] = None
            continue

        match = find_industry_beta(
            industry_betas_df = damo_df,
            company_sector    = None,
            company_industry  = industry_name
        )
        damo_betas[t] = match["beta"] if match else None
    # ─────────────────────────────────────────────────────────────────────────────
        # ─── Maintain a stack of tickers that yielded a Damodaran β ─────────────────
    if "damo_history" not in st.session_state:
        st.session_state["damo_history"] = []

    #  Push any newly‐scored ticker onto the history stack
    for t in sel_tickers:
        b = damo_betas.get(t)
        if b is not None and (not st.session_state["damo_history"]
                            or st.session_state["damo_history"][-1] != t):
            st.session_state["damo_history"].append(t)

    #  Remove any tickers no longer selected
    st.session_state["damo_history"] = [
        t for t in st.session_state["damo_history"] if t in sel_tickers
    ]



    # — 9-3) Combined β-chart (FF-5 & Pure-CAPM vs Damo)
    # Show chart if ANY method has been run, not all

    # Build a simple dict of { ticker: damo_beta } for each selected ticker
    # ─── Damodaran betas per-company ────────────────────────────────

    damo_betas: dict[str, float] = {}

    for t in sel_tickers:
        # 1) figure out which folder the ticker lived in (US, DE, NZ, AU, etc.)
        folder_code = ticker_to_region(t)
        # 2) map that folder to your Damodaran region key
        damo_region = map_folder_to_region(folder_code)
        # 3) fetch the already-working table for that region

        damo_df = st.session_state["damo_industry_df"]


        # 4) find this ticker's industry name (you populated this map earlier)
        industry_name = ticker_industry_map.get(t)
        if industry_name is None:
            damo_betas[t] = None
            continue

        # 5) look it up in the DF
        match = find_industry_beta(
            industry_betas_df = damo_df,
            company_sector    = None,
            company_industry  = industry_name
        )
        damo_betas[t] = match["beta"] if match else None

    has_ff5_data = "FF-5" in methods and st.session_state.get("ff5_ran", False) and betas
    # ─── Build a clean capm dict for the combined β‐chart ─────────────────
    capm: dict[str, dict[str, float]] = {
        t: st.session_state["capm_results"][t]
        for t in sel_tickers
        if t in st.session_state.get("capm_results", {})
    }

    has_capm_data = (
        "CAPM" in methods
        and st.session_state.get("capm_ran", False)
        and bool(capm)
    )

    has_damo_data = (
        "Damo α" in methods
        and st.session_state.get("damo_ran", False)
        and any(damo_betas.values())
    )

    # Auto-run CAPM when selected (replacing the old button behavior)
    if "CAPM" in methods:
        if "capm_results" not in st.session_state:
            st.session_state["capm_results"] = {}
        if "capm_resids" not in st.session_state:
            st.session_state["capm_resids"] = {}

        # Load stock returns for any missing tickers first
        if "stock_returns" not in st.session_state:
            st.session_state["stock_returns"] = {}
        
        for ticker in sel_tickers:
            if ticker not in st.session_state["stock_returns"]:
                csv_path = f"attached_assets/returns_{ticker.replace('.', '_')}.csv"
                if os.path.isfile(csv_path):
                    df_ret = pd.read_csv(csv_path, parse_dates=True, index_col=0)
                    st.session_state["stock_returns"][ticker] = df_ret["Return"]
                else:
                    st.sidebar.error(f"Missing returns file for {ticker}: {csv_path}")

        # Calculate CAPM for each selected ticker
        mkt_returns = st.session_state.get("market_returns", {})
        for ticker in sel_tickers:
            if ticker not in st.session_state.get("capm_results", {}):
                if ticker in st.session_state.get("stock_returns", {}):
                    region = ticker_to_region(ticker)
                    if region in mkt_returns:
                        stock_ret = st.session_state["stock_returns"][ticker]
                        market_ret = mkt_returns[region]
                        st.write(f"DEBUG {ticker} stock_ret.index:", stock_ret.index)
                        st.write(f"DEBUG {ticker} market_ret.index:", market_ret.index)

                        # Calculate CAPM using FF5 data if available 
                        folder = ticker_to_region(ticker)
                        try:
                            ff5_data = get_ff5_data_by_folder(ticker, folder)
                            capm_dict = compute_capm_beta(stock_ret, ff5_data)
                            st.session_state["capm_results"][ticker] = capm_dict
                            st.session_state["capm_ran"] = True # New stocks will have Alphas graphed

                            
                            # Create proper residuals Series with dates
                            if "residuals" in capm_dict and "dates" in capm_dict:
                                dates = pd.to_datetime(capm_dict["dates"])
                                residuals_series = pd.Series(capm_dict["residuals"], index=dates)
                                st.session_state["capm_resids"][ticker] = residuals_series
                                st.sidebar.success(f"✅ CAPM residuals stored for {ticker}: {len(residuals_series)} points")
                            else:
                                st.sidebar.warning(f"⚠️ CAPM dict missing residuals/dates for {ticker}: {list(capm_dict.keys())}")
                        except Exception as e:
                            st.sidebar.error(f"❌ CAPM calculation failed for {ticker}: {e}")
                            # Fallback: just calculate beta without residuals
                            beta = compute_pure_capm_beta(stock_ret, market_ret)
                            st.session_state["capm_results"][ticker] = {"beta": beta}

        st.session_state["capm_ran"] = True

        # Debug: Show what we have in CAPM residuals
        st.sidebar.write(f"DEBUG: CAPM residuals keys: {list(st.session_state.get('capm_resids', {}).keys())}")
        st.sidebar.write(f"DEBUG: Selected tickers: {sel_tickers}")

        # Update the capm variable for the chart
        capm = {
            t: st.session_state["capm_results"][t]
            for t in sel_tickers
            if t in st.session_state["capm_results"]
        }
        has_capm_data = "CAPM" in methods and st.session_state.get("capm_ran", False) and bool(st.session_state.get("capm_results", {}))

        has_damo_data = "Damo α" in methods and st.session_state.get("damo_ran", False) and any(damo_betas.values())

        
        has_damo_data = (
            "Damo α" in methods
            and st.session_state.get("damo_ran", False)
            and any(damo_betas.values())
        )

    with st.expander("🔎 CAPM Debugger (click to expand)", expanded=False):
        st.write("has_capm_data:", has_capm_data, "capm:", capm)


    if has_ff5_data or has_capm_data or has_damo_data:
        st.markdown("#### Factor Betas: FF-5 vs Pure-CAPM vs Damo")
        fig = go.Figure()

        # ─ FF-5 traces (only if FF-5 was clicked and ran)
        if has_ff5_data:
            for t, bdict in betas.items():
                fig.add_trace(go.Scatter(
                    x=list(bdict.keys()),
                    y=list(bdict.values()),
                    mode="lines+markers",
                    name=f"{t} (FF-5)",
                    line=dict(color=color_map[t]),
                    marker=dict(symbol="circle", color=color_map[t]),
                ))
        
    
        # — Pure-CAPM traces (only if CAPM was clicked and ran) —
        if has_capm_data:
            for t, cres in capm.items():
                fig.add_trace(go.Scatter(
                    x=list(cres.keys()),            # This generalizes in case you ever add more factors
                    y=list(cres.values()),          # My β value(s) for each factor
                    mode="lines+markers",
                    name=f"{t} (CAPM)",
                    line=dict(color=color_map[t], dash="dot"),
                    marker=dict(symbol="x", size=12, color=color_map[t]),
                ))


    # — Damodaran β traces (only when selected in methods and checkbox is checked) —
        if has_damo_data and show_damo:
            for t, b in damo_betas.items():
                if b is not None:
                    fig.add_trace(go.Scatter(
                        x=["Damodaran β"],
                        y=[b],
                        mode="markers",
                        name=f"{t} (Damo β)",
                        marker=dict(symbol="diamond", size=12, color=color_map[t]),
                    ))

        fig.update_layout(
            xaxis_title="Factor / Industry",
            yaxis_title="β Coefficient",
            legend_title="Ticker & Model",
        )

        st.plotly_chart(fig, use_container_width=True, key="beta_chart")

    # — (9-5) Compute & plot Damodaran α series  -----------------------
    if sel_tickers and "Damo" in methods:
        # Grab our saved per-ticker returns dict
        stock_returns = st.session_state.get("stock_returns", {})

        # Build per-ticker residuals dict
        industry_resids: dict[str, pd.Series] = {}
        for t in sel_tickers:
            b = damo_betas.get(t)
            if b is None:
                continue

            # Map ticker to its region’s market returns
            region = ticker_to_region(t)
            mkt = mkt_returns.get(region)
            if mkt is None:
                continue

            # Daily risk-free (assuming rf_annual defined earlier)
            rf_daily = rf_annual / 252

            # Fetch the stock’s daily returns series
            stock_ret = stock_returns.get(t)
            if stock_ret is None:
                continue

            # Compute Damodaran residuals (α = actual − β·market excess)
            res = compute_industry_residuals(
                stock_ret=stock_ret,
                market_ret=mkt,
                beta=b,
                risk_free_rate=rf_daily
            )
            industry_resids[t] = res

        # Now plot them
        if "Damo α" in methods and industry_resids:
            st.markdown("#### 🏭 Damodaran-Model α")
            fig_damo_alpha = go.Figure()
            for t, resid in industry_resids.items():
                fig_damo_alpha.add_trace(
                    go.Scatter(
                        x=resid.index,
                        y=resid.values,
                        mode="lines",
                        name=f"{t} (Damo α)"
                    )
                )

            # Zoom into the date-range chosen by your α slider
            fig_damo_alpha.update_layout(
                xaxis=dict(range=[f"{alpha_start}-01", f"{alpha_end}-12"]),
                height=350,
                legend_title="Ticker & Model"
            )

            st.plotly_chart(fig_damo_alpha, use_container_width=True, key="damo_alpha_chart")
            # ─── Persist these for the combined chart ─────────────────
            st.session_state["damo_resids"] = industry_resids


    # — 11) Plot FF-5 residuals (pure α) over time ——————————
        if "ff5_resids" in st.session_state and sel_tickers:
            alpha_dict = st.session_state["ff5_resids"]

            # Turn dict of Series into one DataFrame
            # ─── Turn dict of Series into one DataFrame ────────────
            df_alpha = pd.DataFrame(alpha_dict)

            # Coerce whatever your index is into real timestamps
            df_alpha.index = pd.to_datetime(df_alpha.index, errors="coerce")
            df_alpha.index.name = "Date"
            # Build monthly returns DF to overlay on same chart —
            stock_rets = st.session_state.get("stock_returns", {})
            monthly_dict = {}
            for t in sel_tickers:
                if t in stock_rets:
                    # Compute a proper Series with DateTimeIndex
                    s = (1 + stock_rets[t]).resample("M").prod() - 1
                    # Convert that to period‐start timestamp
                    if hasattr(s.index, 'to_period'):
                        s.index = s.index.to_period("M").to_timestamp()
                    monthly_dict[t] = s

            df_ret = pd.DataFrame(monthly_dict)

            fig_alpha = go.Figure()

        # 1) If FF-5 ran, plot FF-5 alpha (dot)
            if st.session_state.get("ff5_ran", False):
                df_ff5 = pd.DataFrame(st.session_state["ff5_resids"])
                df_ff5.index = pd.to_datetime(df_ff5.index)
                for t in sel_tickers:
                    if t in df_ff5:
                        fig_alpha.add_trace(go.Scatter(
                            x=df_ff5.index,
                            y=df_ff5[t],
                            mode="lines",
                            name=f"{t} FF5 α",
                            line=dict(color=color_map[t], dash="dot"),
                        ))


            # 3) If CAPM ran, plot CAPM alpha (solid)
            st.write("capm_ran =", st.session_state.get("capm_ran"))
            st.write("resids keys:", list(st.session_state.get("capm_resids", {}).keys()))


            if st.session_state.get("capm_ran", False):
                df_capm = pd.DataFrame(st.session_state["capm_resids"])
                df_capm.index = pd.to_datetime(df_capm.index)


                for t in sel_tickers:
                    if t in df_capm:
                        fig_alpha.add_trace(go.Scatter(
                            x=df_capm.index,
                            y=df_capm[t],
                            mode="lines",
                            name=f"{t} CAPM α",
                            line=dict(color=color_map[t], dash="solid"),
                        ))


    # ─── Plot FF-5, CAPM & Damodaran Residuals Over Time vs Returns ───────────────

    if sel_tickers :
        
        # Load stock returns into session state if not already loaded
        stock_rets = st.session_state.get("stock_returns", {})
        for ticker in sel_tickers:
            if ticker not in stock_rets:
                
                fn = f"attached_assets/returns_{ticker.replace('.', '_')}.csv"
                if os.path.isfile(fn):
                    df_ret = pd.read_csv(fn, parse_dates=True, index_col=0)
                    stock_rets[ticker] = df_ret["Return"]
                else:
                    st.sidebar.error(f"Missing returns file for {ticker}: {fn}")
        st.session_state["stock_returns"] = stock_rets

        st.markdown("#### 📈 Model Residuals (Alphas) Over Time vs Returns")  
        # 1) Build one dict of all residual Series
        alpha_dict: dict[str, pd.Series] = {}

        # FF-5
        if "FF-5" in methods and st.session_state.get("ff5_ran", False):
            alpha_dict.update(st.session_state["ff5_resids"])

        # CAPM
        if "CAPM" in methods and st.session_state.get("capm_ran", False):
            alpha_dict.update(st.session_state["capm_resids"])

        # Damodaran - only if it was actually run
        if "Damo α" in methods and st.session_state.get("damo_ran", False):
            damo_resids_existing = st.session_state.get("damo_resids", {})
            alpha_dict.update(damo_resids_existing)

        # 2) Make a DataFrame, parse dates, clip to slider range
        df_alpha = pd.DataFrame(alpha_dict)
        df_alpha.index = pd.to_datetime(df_alpha.index)
        df_alpha.index.name = "Date"
        df_alpha = df_alpha.loc[f"{alpha_start}-01-01": f"{alpha_end}-12-31"]

        # 3) Build monthly returns from session state
        stock_rets = st.session_state.get("stock_returns", {})
        monthly = {
            f"{t} Return": (1 + stock_rets[t]).resample("M").prod().rename(f"{t} Return")
            for t in sel_tickers if t in stock_rets
        }
        df_ret = pd.DataFrame(monthly)
        if not df_ret.empty and hasattr(df_ret.index, 'to_period'):
            df_ret.index = df_ret.index.to_period("M").to_timestamp()
            df_ret = df_ret.loc[df_alpha.index.min(): df_alpha.index.max()]

        # 4) Plot all traces in the *sidebar order* of methods
        fig = go.Figure()
        for model in methods:
            if model == "FF-5" and st.session_state.get("ff5_ran", False):
                ff5_resids = st.session_state.get("ff5_resids", {})
                for t in sel_tickers:
                    if t in ff5_resids:
                        # Get the time-filtered series
                        series = ff5_resids[t]
                        series.index = pd.to_datetime(series.index)
                        filtered_series = series.loc[f"{alpha_start}-01-01": f"{alpha_end}-12-31"]
                        if not filtered_series.empty:
                            fig.add_trace(go.Scatter(
                                x=filtered_series.index,
                                y=filtered_series.values,
                                mode="lines",
                                name=f"{t} FF-5 α",
                                line=dict(color=color_map[t], dash="dash"),
                            ))

            if model == "CAPM" and st.session_state.get("capm_ran", False):
                capm_resids = st.session_state.get("capm_resids", {})
                for t in sel_tickers:
                    if t in capm_resids:
                        # Get the time-filtered series
                        series = capm_resids[t]
                        series.index = pd.to_datetime(series.index)
                        filtered_series = series.loc[f"{alpha_start}-01-01": f"{alpha_end}-12-31"]
                        if not filtered_series.empty:
                            fig.add_trace(go.Scatter(
                                x=filtered_series.index,
                                y=filtered_series.values,
                                mode="lines",
                                name=f"{t} CAPM α",
                                line=dict(color=color_map[t], dash="dot"),
                            ))

            if model == "Damo α" and st.session_state.get("damo_ran", False):
                damo_resids_stored = st.session_state.get("damo_resids", {})
                for name, series in damo_resids_stored.items():
                    ticker = name.split()[0]
                    # Apply time filtering to Damodaran data too
                    series.index = pd.to_datetime(series.index)
                    filtered_series = series.loc[f"{alpha_start}-01-01": f"{alpha_end}-12-31"]
                    if not filtered_series.empty:
                        fig.add_trace(go.Scatter(
                            x=filtered_series.index,
                            y=filtered_series.values,
                            mode="lines",
                            name=name,
                            line=dict(color=color_map[ticker], dash="dashdot"),
                        ))

        # 5) Overlay returns with same time filtering as alphas
        stock_rets = st.session_state.get("stock_returns", {})
        for t in sel_tickers:
            if t in stock_rets:
                # Apply the same time filtering to returns
                series = stock_rets[t]
                series.index = pd.to_datetime(series.index)
                # Convert to monthly returns and apply time filter
                monthly_returns = (1 + series).resample("M").prod() - 1
                if hasattr(monthly_returns.index, 'to_period'):
                    monthly_returns.index = monthly_returns.index.to_period("M").to_timestamp()
                
                filtered_returns = monthly_returns.loc[f"{alpha_start}-01-01": f"{alpha_end}-12-31"]
                if not filtered_returns.empty:
                    fig.add_trace(go.Scatter(
                        x=filtered_returns.index,
                        y=filtered_returns.values,
                        mode="lines",
                        name=f"{t} Return",
                        line=dict(color=color_map[t], dash="solid"),
                        opacity=0.7
                    ))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Value",
            legend_title="Series",
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True, key="model_residuals_chart")


    # ─── 9.x) Damodaran betas: download & cache regional Excel ───────────────────────
    if "damo_files" not in st.session_state:
        # Download each region’s file into ./Damodaran/
        damo_files = {
            reg: _download_region_beta(reg)
            for reg in ("US", "Europe", "AU_NZ")
        }
        st.session_state["damo_files"] = damo_files

    # Read them into DataFrames, using xlrd for .xls files

    damo_dfs = {}
    for reg, path in st.session_state["damo_files"].items():
        ext = Path(path).suffix.lower()
        if ext == ".xls":
            # let Pandas use xlrd
            damo_dfs[reg] = pd.read_excel(path, engine="xlrd")
        else:
            # newer workbooks, if any
            damo_dfs[reg] = pd.read_excel(path, engine="openpyxl")


##### WACC START #####
    if sel_tickers:

        # Pull saved model results + flags
        capm_dict = st.session_state.get("capm_results", {})
        capm_ran  = st.session_state.get("capm_ran",  False)
        ff5_ran   = st.session_state.get("ff5_ran",   False)
        damo_ran  = st.session_state.get("damo_ran",  False)

        # Load the Damodaran industry-beta DataFrame (must have been fetched earlier)
        damo_df   = st.session_state.get("damo_industry_df", pd.DataFrame())

        # Annualize RF & market premium from FF-5 data (identical for all models)
        sample_ff5 = next(iter(st.session_state.get("ff5", {}).values()), None)
        if sample_ff5 is not None:
            sample_ff5     = sample_ff5.apply(pd.to_numeric, errors="coerce")
            monthly_rf     = sample_ff5["RF"].mean()    / 100
            monthly_mktrf  = sample_ff5["Mkt-RF"].mean() / 100
            rf_annual      = (1 + monthly_rf)**12 - 1
            mktprem_annual = (1 + monthly_mktrf)**12 - 1
        else:
            rf_annual = mktprem_annual = 0.0

        wacc_rows = []

        for t in sel_tickers:
            # ─── 1) Pick betas ───────────────────────────────────────────────
            beta_capm = capm_dict.get(t, {}).get("beta", 0.0)    if capm_ran else 0.0
            beta_ff5  = betas.get(t, {}).get("Mkt-RF", 0.0)     if ff5_ran  else 0.0

            # ─── 2) Cost of equity ──────────────────────────────────────────
            re_capm = rf_annual + beta_capm * mktprem_annual
            re_ff5  = rf_annual + beta_ff5  * mktprem_annual

            # ─── 3) Fetch that year’s financials ────────────────────────────
            sub = df.query("Ticker == @t and Year == @sel_year")
            if sub.empty:
                continue
            row = sub.iloc[0]

            # ─── 4) After-tax cost of debt ───────────────────────────────────
            rd_at = (abs(row["InterestExpense"]) / row["Debt"]) * (1 - row["tax_rate"]) \
                    if row["Debt"] > 0 else 0.0

            # ─── 5) Capital structure weights ───────────────────────────────
            netD     = row["Debt"] - row["Cash"]
            E_val    = row["EV"] - netD
            tot      = E_val + netD or 1.0
            w_e, w_d = E_val / tot, netD / tot

            # ─── 5a) Pure-CAPM & FF-5 WACC ─────────────────────────────────
            wacc_capm = w_e * re_capm + w_d * rd_at
            wacc_ff5  = w_e * re_ff5  + w_d * rd_at

            # ─── 5b) Damodaran lookup & WACC ────────────────────────────────
            industry    = ticker_industry_map.get(t)
            damo_match  = None
            if damo_ran and not damo_df.empty and industry:
                region_key = ticker_to_region(t)
                damo_match = find_industry_beta(
                    damo_df,
                    company_sector   = region_key,
                    company_industry = industry
                )
            industry_beta = damo_match["beta"] if damo_match else None

            damo_res = (
                calculate_wacc_with_industry_beta(
                    market_beta    = beta_capm,
                    industry_beta  = industry_beta,
                    risk_free_rate = rf_annual,
                    market_premium = mktprem_annual,
                    debt_ratio     = w_d,
                    cost_of_debt   = rd_at,
                    tax_rate       = row["tax_rate"],
                )
                if industry_beta is not None else {}
            # if ("Damo α" in methods and industry_beta is not None) else {}
            )
            re_damo   = damo_res.get("industry_wacc", {}).get("cost_of_equity")
            wacc_damo = damo_res.get("industry_wacc", {}).get("wacc")

            # ─── 6) Append our row ───────────────────────────────────────────
            wacc_rows.append({
                "Ticker":        t,
                "Re (CAPM %)":   f"{re_capm*100:.2f}"    if capm_ran  else "0.00",
                "Re (FF5 %)":    f"{re_ff5*100:.2f}"     if ff5_ran   else "0.00",
                "Re (Damo %)":   f"{re_damo*100:.2f}"    if re_damo    is not None else "n/a",
                "Rd (%)":        f"{rd_at*100:.2f}",
                "wE (%)":        f"{w_e*100:.1f}",
                "wD (%)":        f"{w_d*100:.1f}",
                "WACC (CAPM %)": f"{wacc_capm*100:.2f}"  if capm_ran  else "0.00",
                "WACC (FF5 %)":  f"{wacc_ff5*100:.2f}"   if ff5_ran   else "0.00",
                "WACC (Damo %)": f"{wacc_damo*100:.2f}"  if wacc_damo is not None else "n/a",
            })

        # ─── Render the table (always at the bottom) ─────────────────────────────
        wacc_df = pd.DataFrame(wacc_rows).set_index("Ticker")
        st.markdown("#### 🧮 WACC by Company")
##### WACC END #####
        display_cols = []
        if "CAPM" in methods and capm_ran:
            display_cols += ["Re (CAPM %)", "WACC (CAPM %)"]
        if "FF-5" in methods and ff5_ran:
            display_cols += ["Re (FF5 %)",  "WACC (FF5 %)"]
        if "Damo α" in methods and damo_ran:
            display_cols += ["Re (Damo %)",  "WACC (Damo %)"]

        display_cols += ["Rd (%)", "wE (%)", "wD (%)"]

        st.dataframe(wacc_df[display_cols])
        st.session_state["last_year_wacc"] = sel_year
        ######################################DEBUG#########################
        # Calculate Intrinsic Value & Cash Flow Summary
        fcff_data = []

        for ticker in sel_tickers:
            t_rows = df[df['Ticker'] == ticker]
            if t_rows.empty:
                fcff_data.append({
                    "Ticker": ticker,
                    "FCFF (Yr N)": "-",
                    "FCFE (Yr N)": "-",
                    "Sales": "-",            # ← Sales when no data
                    "Terminal Value": "-",
                    "Shares Outstanding": "-",
                    "Intrinsic Value (CAPM)": "-",
                    "Intrinsic Value (FF5)": "-",
                    "Intrinsic Value (Damo)": "-",
                })
                continue

            # Use selected year, or latest if not present
            # Only include companies that have data for the selected year
            if sel_year not in t_rows['Year'].values:
                continue  # Skip this ticker if it doesn't have data for selected year
            row = t_rows[t_rows['Year'] == sel_year].iloc[0]

            ebit = row["EBIT"]
            capex = row["CapEx"]
            nwc = row["ChangeNWC"]
            tax = row["CashTaxesPaid"]
            interest = row.get("InterestExpense", 0)
            debt = row.get("Debt", 0)
            sales = row["Sales"]


            # FCFF
            ebit = row["EBIT"]
            tax_rate = row["tax_rate"]
            dep_amort = row["DepreciationAndAmortization"]
            capex = row["CapEx"]
            delta_nwc = row["ChangeNWC"]

            fcff = ebit * (1 - tax_rate) + dep_amort - capex - delta_nwc

            # Filter out Terminal Value rows before sorting, then sort by numeric years
            t_rows_numeric = t_rows[t_rows["Year"] != "Terminal Value"].copy()
            t_rows_numeric["Year"] = pd.to_numeric(t_rows_numeric["Year"], errors='coerce')
            t_rows_sorted = t_rows_numeric.sort_values("Year")
            debt_idx = t_rows_sorted[t_rows_sorted['Year'] == row['Year']].index[0]
            if debt_idx > 0:
                debt_prev = t_rows_sorted.iloc[t_rows_sorted.index.get_loc(debt_idx) - 1]["Debt"]
            else:
                debt_prev = 0
            delta_debt = debt - debt_prev

            # FCFE

            interest = row.get("InterestExpense", 0)
            debt = row.get("Debt", 0)
            fcfe = fcff - (interest * (1 - tax_rate)) + delta_debt

            fcff_data.append({
                "Ticker": ticker,
                "FCFF (Yr N)": f"{fcff:,.0f}",
                "FCFE (Yr N)": f"{fcfe:,.0f}",
                "Sales (M)":      f"{sales:,.0f}",
                "Terminal Value (M)": "-",          # To be filled in next step
                "Shares Outstanding (M)": "-",      # To be filled in next step
                "Intrinsic Value (CAPM)": "-",  # To be filled in later
                "Intrinsic Value (FF5)": "-",
                "Intrinsic Value (Damo)": "-",
            })

        # fcff_table = pd.DataFrame(fcff_data)


        #  Then display the table
        fcff_table = pd.DataFrame(fcff_data)
        
        # 1) Pull Terminal Values out of your main `df` (I appended these earlier)
        tv_series = (
            df[df["Year"] == "Terminal Value"]
            .set_index("Ticker")["FCFF"]
        )

        # 2) Map them into the summary table (leaving "-" where no TV exists)
        fcff_table["Terminal Value (M)"] = (
            fcff_table["Ticker"]
            .map(lambda ticker: f"{tv_series.get(ticker, 0):.3f}" if pd.notna(tv_series.get(ticker)) else "-")
        )

        # 3) Re-add the placeholder columns so the layout matches your original
        
        # Get actual shares outstanding from the data
        for idx, row_data in fcff_table.iterrows():
            ticker = row_data["Ticker"]
            ticker_data = df[df["Ticker"] == ticker]
            if not ticker_data.empty:
                # Get actual shares outstanding from the data (cleaner approach)
                shares_data = df[df["Year"] == sel_year].set_index("Ticker")["SharesOutstanding"]
                fcff_table["Shares Outstanding (M)"] = fcff_table["Ticker"].map(
                lambda ticker: f"{shares_data.get(ticker, 0):,.0f}" 
                if pd.notna(shares_data.get(ticker)) and shares_data.get(ticker) != 0 
                else "-"
        )
      

        #####################################################
        # ── 3.1) Grab Year‑N raw metrics for each ticker ──────────────────────────────
        df_year = df[df["Year"] == sel_year].set_index("Ticker")
        ev_series       = df_year["EV"]
        ebitda_series   = df_year["EBITDA"]
        net_debt_series = df_year["Debt"] - df_year["Cash"]
        shares_series   = df_year["SharesOutstanding"]  # or whatever your raw df column is named

        # ── 3.2) Map into the summary table ─────────────────────────────────────────
        fcff_table["Implied EV"] = fcff_table["Ticker"].map(lambda t: ev_series.get(t, pd.NA))
        fcff_table["Implied Equity"] = fcff_table["Ticker"].map(
            lambda t: ev_series.get(t, pd.NA) - net_debt_series.get(t, pd.NA)
        )
        fcff_table["Implied Price/Share"] = fcff_table["Ticker"].map(
            lambda t: (
                (ev_series.get(t, pd.NA) - net_debt_series.get(t, pd.NA))
                / shares_series.get(t, pd.NA)
            ) if pd.notna(shares_series.get(t, pd.NA)) and shares_series.get(t, pd.NA) != 0 else pd.NA
        )

        ######################################################
        # 4) Only display table when all estimation methods are selected
        required_methods = {"CAPM", "FF-5", "Damo α"}
        if required_methods.issubset(set(methods)) and capm_ran and ff5_ran and damo_ran:

        # 1) Heavy LLM/DCF work inside the spinner
            with st.spinner("🧮 Calculating intrinsic‑value tables… this may take 10–20 s."):
                fcff_table = calculate_all_intrinsic_values(fcff_table, wacc_df, df)

            # 2) Once the context ends, the spinner auto‑disappears,
            #    and you render your summary immediately below:
            st.markdown("### 🪙 Intrinsic Value & Cash Flow Summary")

            #########################-------->####Arrow Sign########
            st.markdown(
            """
            <p style="text-align:right; font-size:0.85rem; color:#888;">
            ⇆ <em>slide&nbsp;→</em> to view “Implied Price / Share”
            </p>
            """,
            unsafe_allow_html=True,
            )
            #######################################################
            st.dataframe(fcff_table, hide_index=True)

        # THE CHART CODE 
            # Intrinsic value visualization chart
            st.markdown("### 📊 Intrinsic Value Comparison by WACC Method")
            
            # Create intrinsic value chart
            fig_intrinsic = go.Figure()
            
            # Extract intrinsic values for each ticker and method
            for _, row in fcff_table.iterrows():
                ticker = row["Ticker"]
                
                # Parse intrinsic values (remove commas and convert to float)
                try:
                    capm_value = float(str(row["Intrinsic Value (CAPM)"]).replace(",", "")) if row["Intrinsic Value (CAPM)"] != "-" else None
                    ff5_value = float(str(row["Intrinsic Value (FF5)"]).replace(",", "")) if row["Intrinsic Value (FF5)"] != "-" else None
                    damo_value = float(str(row["Intrinsic Value (Damo)"]).replace(",", "")) if row["Intrinsic Value (Damo)"] != "-" else None
                except (ValueError, TypeError):
                    continue
                    
                # Get the color for this ticker
                ticker_color = color_map.get(ticker, "#636EFA")  # Default color if not found
                
                # Add traces for each method
                methods_data = [
                    ("CAPM", capm_value, "circle"),
                    ("FF5", ff5_value, "square"), 
                    ("Damo", damo_value, "diamond")
                ]
                
                for method_name, value, symbol in methods_data:
                    if value is not None:
                        fig_intrinsic.add_trace(go.Scatter(
                            x=[method_name],
                            y=[value],
                            mode="markers",
                            name=f"{ticker}",
                            marker=dict(
                                symbol=symbol,
                                size=12,
                                color=ticker_color,
                                line=dict(width=2, color="white")
                            ),
                            showlegend=method_name == "CAPM",  # Only show legend for first method per ticker
                            legendgroup=ticker  # Group all methods for same ticker
                        ))
            
            fig_intrinsic.update_layout(
                xaxis_title="WACC Method",
                yaxis_title="Intrinsic Value per Share ($)",
                legend_title="Stock Ticker",
                template="plotly_dark",
                height=400,
                xaxis=dict(categoryorder="array", categoryarray=["CAPM", "FF5", "Damo"])
            )
            
            st.plotly_chart(fig_intrinsic, use_container_width=True, key="intrinsic_value_chart")
            st.markdown(
            """
            All figures are reported in local currency (millions) for each company.<br>
            *FF-5 factor betas data courtesy of the [Kenneth R. French Data Library](
            https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/index.html).*

            Source code at Github: https://github.com/tomektomeknyc/dcf

            """,
            unsafe_allow_html=True,
    )
with tab_desc:
    render_project_description_tab()

with tab_qa:
    render_qa_tab()

with tab_fin_report:
    render_fin_report_tab(selected_stocks)

   
