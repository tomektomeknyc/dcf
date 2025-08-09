#fin_stat_downloader.py

from __future__ import annotations

import io
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
import yfinance as yf

# ───────────────────────────── Arrow-safe display helpers ─────────────────────────────

def arrow_friendly(df: pd.DataFrame) -> pd.DataFrame:
    """Make df safe for Streamlit/Arrow: numeric where possible, else string. Also stringify columns."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = df.columns.map(lambda c: str(c))
    for c in df.columns:
        try:
            num = pd.to_numeric(df[c], errors="coerce")
            if (num.notna().mean() >= 0.7):
                df[c] = num
            else:
                df[c] = df[c].astype("string")
        except Exception:
            df[c] = df[c].astype("string")
    df.index = df.index.astype("string")
    return df

def show_df(df: pd.DataFrame, **kwargs):
    st.dataframe(arrow_friendly(df), **kwargs)

# ───────────────────────────── Label sets (for Selected sheets) ───────────────────────

IS_SELECTED = [
    "Total Revenue", "Cost Of Revenue", "Gross Profit",
    "Operating Income", "Net Income", "Diluted EPS", "Interest Expense",
]
BS_SELECTED = [
    "Total Assets", "Total Liabilities", "Total Equity",
    "Total Current Assets", "Total Current Liabilities",
    "Cash & Equivalents", "Total Debt",
]
CF_SELECTED = [
    "Operating Cash Flow", "Capital Expenditure",
    "Free Cash Flow", "Depreciation & Amortization",
]

# ───────────────────────────── Utilities ──────────────────────────────────────────────

def _norm_ticker_for_yf(user_ticker: str) -> str:
    """
    Convert common RIC-like inputs to Yahoo symbols.
    Examples:
      ADSK.O -> ADSK
      AAPL.O -> AAPL
      AIR.NZ -> AIR.NZ  (keep)
    """
    s = (user_ticker or "").upper().strip()
    if "." in s:
        if s.endswith(".O") or s.endswith(".N") or s.endswith(".K"):
            return s.split(".")[0]
        return s
    return s

def _year_from_col(col) -> Optional[str]:
    try:
        return str(pd.to_datetime(col).year)
    except Exception:
        m = re.match(r"(\d{4})", str(col))
        return m.group(1) if m else None

def _limit_years(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    cols = [c for c in df.columns if str(c).isdigit()]
    cols = sorted(cols, key=lambda x: int(x))
    take = cols[-n:] if len(cols) > n else cols
    return df.loc[:, take]

def _limit_quarters(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df.iloc[:, -n:] if df.shape[1] > n else df

# ───────────────────────────── Yahoo fetchers ────────────────────────────────────────

def _yf_statement(tkr: str, which: str) -> pd.DataFrame:
    t = yf.Ticker(tkr)
    if which == "financials":
        raw = t.financials
    elif which == "balance_sheet":
        raw = t.balance_sheet
    elif which == "cashflow":
        raw = t.cashflow
    elif which == "quarterly_financials":
        raw = t.quarterly_financials
    elif which == "quarterly_balance_sheet":
        raw = t.quarterly_balance_sheet
    elif which == "quarterly_cashflow":
        raw = t.quarterly_cashflow
    else:
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    raw = raw.copy()
    if which.startswith("quarterly"):
        raw.columns = [pd.to_datetime(c).strftime("%Y-%m") if not isinstance(c, str) else c for c in raw.columns]
        raw = raw.loc[:, ~pd.Index(raw.columns).duplicated(keep="last")]
        return raw

    cols = []
    for c in raw.columns:
        y = _year_from_col(c)
        cols.append(y if y else str(c))
    raw.columns = cols
    raw = raw.loc[:, ~pd.Index(raw.columns).duplicated(keep="last")]
    return raw

def fetch_yahoo_all(tkr: str) -> Dict[str, pd.DataFrame]:
    return {
        "IS_A": _yf_statement(tkr, "financials"),
        "BS_A": _yf_statement(tkr, "balance_sheet"),
        "CF_A": _yf_statement(tkr, "cashflow"),
        "IS_Q": _yf_statement(tkr, "quarterly_financials"),
        "BS_Q": _yf_statement(tkr, "quarterly_balance_sheet"),
        "CF_Q": _yf_statement(tkr, "quarterly_cashflow"),
    }

# ───────────────────────────── Alpha Vantage backfill (optional) ─────────────────────

def _av_get(fn: str, symbol: str, api_key: str) -> Optional[dict]:
    url = "https://www.alphavantage.co/query"
    params = {"function": fn, "symbol": symbol, "apikey": api_key}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        js = r.json()
        if "Note" in js or "Information" in js:
            return None
        return js
    except Exception:
        return None

def _to_num(x):
    try:
        return float(x)
    except Exception:
        return None

def fetch_alpha_vantage_annual(symbol: str, api_key: str) -> Dict[str, pd.DataFrame]:
    out = {"IS_A": pd.DataFrame(), "BS_A": pd.DataFrame(), "CF_A": pd.DataFrame()}
    if not api_key:
        return out

    js_is = _av_get("INCOME_STATEMENT", symbol, api_key)
    if js_is and "annualReports" in js_is:
        tmp = {}
        for rpt in js_is["annualReports"]:
            y = _year_from_col(rpt.get("fiscalDateEnding", ""))
            if not y: continue
            tmp.setdefault("Total Revenue", {})[y] = _to_num(rpt.get("totalRevenue"))
            tmp.setdefault("Cost Of Revenue", {})[y] = _to_num(rpt.get("costOfRevenue"))
            tmp.setdefault("Gross Profit", {})[y] = _to_num(rpt.get("grossProfit"))
            tmp.setdefault("Operating Income", {})[y] = _to_num(rpt.get("operatingIncome"))
            tmp.setdefault("Net Income", {})[y] = _to_num(rpt.get("netIncome"))
            tmp.setdefault("Diluted EPS", {})[y] = _to_num(rpt.get("epsdiluted"))
            tmp.setdefault("Interest Expense", {})[y] = _to_num(rpt.get("interestExpense"))
        if tmp: out["IS_A"] = pd.DataFrame(tmp).T

    js_bs = _av_get("BALANCE_SHEET", symbol, api_key)
    if js_bs and "annualReports" in js_bs:
        tmp = {}
        for rpt in js_bs["annualReports"]:
            y = _year_from_col(rpt.get("fiscalDateEnding", ""))
            if not y: continue
            tmp.setdefault("Total Assets", {})[y] = _to_num(rpt.get("totalAssets"))
            tmp.setdefault("Total Liabilities", {})[y] = _to_num(rpt.get("totalLiabilities"))
            tmp.setdefault("Total Equity", {})[y] = _to_num(rpt.get("totalShareholderEquity"))
            tmp.setdefault("Total Current Assets", {})[y] = _to_num(rpt.get("totalCurrentAssets"))
            tmp.setdefault("Total Current Liabilities", {})[y] = _to_num(rpt.get("totalCurrentLiabilities"))
            tmp.setdefault("Cash & Equivalents", {})[y] = _to_num(rpt.get("cashAndCashEquivalentsAtCarryingValue") or rpt.get("cashAndShortTermInvestments"))
            total_debt = _to_num(rpt.get("shortLongTermDebtTotal") or rpt.get("shortTermDebt")) or 0
            lt_debt = _to_num(rpt.get("longTermDebt")) or 0
            val = (total_debt + lt_debt) if (total_debt or lt_debt) else None
            tmp.setdefault("Total Debt", {})[y] = val
        if tmp: out["BS_A"] = pd.DataFrame(tmp).T

    js_cf = _av_get("CASH_FLOW", symbol, api_key)
    if js_cf and "annualReports" in js_cf:
        tmp = {}
        for rpt in js_cf["annualReports"]:
            y = _year_from_col(rpt.get("fiscalDateEnding", ""))
            if not y: continue
            ocf = _to_num(rpt.get("operatingCashflow"))
            capex = _to_num(rpt.get("capitalExpenditures"))
            depam = _to_num(rpt.get("depreciationAndAmortization"))
            fcf = (ocf if ocf is not None else 0) + (capex if capex is not None else 0)
            tmp.setdefault("Operating Cash Flow", {})[y] = ocf
            tmp.setdefault("Capital Expenditure", {})[y] = capex
            tmp.setdefault("Free Cash Flow", {})[y] = fcf if (ocf is not None or capex is not None) else None
            tmp.setdefault("Depreciation & Amortization", {})[y] = depam
        if tmp: out["CF_A"] = pd.DataFrame(tmp).T

    return out

# ───────────────────────────── Merge & transforms ─────────────────────────────────────

def _map_selected(full_df: pd.DataFrame, mapping: Dict[str, List[str]], labels: List[str]) -> pd.DataFrame:
    if full_df is None or full_df.empty:
        return pd.DataFrame(index=labels)
    out = pd.DataFrame(index=labels, columns=full_df.columns, dtype="float64")
    for label, candidates in mapping.items():
        for cand in candidates:
            if cand in full_df.index:
                out.loc[label] = pd.to_numeric(full_df.loc[cand], errors="coerce")
                break
    return out

def build_selected_tables(yahoo: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    is_map = {
        "Total Revenue": ["Total Revenue", "TotalRevenue"],
        "Cost Of Revenue": ["Cost Of Revenue", "CostOfRevenue"],
        "Gross Profit": ["Gross Profit", "GrossProfit"],
        "Operating Income": ["Operating Income", "OperatingIncome"],
        "Net Income": ["Net Income", "NetIncome"],
        "Diluted EPS": ["Diluted EPS", "DilutedEPS"],
        "Interest Expense": ["Interest Expense", "InterestExpense"],
    }
    bs_map = {
        "Total Assets": ["Total Assets", "TotalAssets"],
        "Total Liabilities": ["Total Liabilities Net Minority Interest", "Total Liabilities", "TotalLiab"],
        "Total Equity": ["Total Equity Gross Minority Interest", "Total Stockholder Equity", "TotalEquityGrossMinorityInterest", "TotalStockholderEquity"],
        "Total Current Assets": ["Total Current Assets", "TotalCurrentAssets"],
        "Total Current Liabilities": ["Total Current Liabilities", "TotalCurrentLiabilities"],
        "Cash & Equivalents": ["Cash And Cash Equivalents", "CashAndCashEquivalents", "CashCashEquivalentsAndShortTermInvestments"],
        "Total Debt": ["Total Debt", "TotalDebt"],
    }
    cf_map = {
        "Operating Cash Flow": ["Operating Cash Flow", "OperatingCashFlow"],
        "Capital Expenditure": ["Capital Expenditure", "CapitalExpenditure"],
        "Free Cash Flow": ["Free Cash Flow", "FreeCashFlow"],
        "Depreciation & Amortization": ["Depreciation", "DepreciationAmortizationDepletion", "DepreciationAndAmortization"],
    }

    is_sel = _map_selected(yahoo["IS_A"], is_map, IS_SELECTED)
    bs_sel = _map_selected(yahoo["BS_A"], bs_map, BS_SELECTED)
    cf_sel = _map_selected(yahoo["CF_A"], cf_map, CF_SELECTED)

    return {"IS_A_SEL": is_sel, "BS_A_SEL": bs_sel, "CF_A_SEL": cf_sel}

def merge_annual_yahoo_alpha(yahoo: Dict[str, pd.DataFrame], av: Dict[str, pd.DataFrame], years: int) -> Dict[str, pd.DataFrame]:
    out = {}
    for key_y, key_av in (("IS_A", "IS_A"), ("BS_A", "BS_A"), ("CF_A", "CF_A")):
        ydf = yahoo.get(key_y, pd.DataFrame())
        adf = av.get(key_av, pd.DataFrame())
        if ydf is None or ydf.empty:
            merged = adf
        elif adf is None or adf.empty:
            merged = ydf
        else:
            merged = ydf.copy()
            for col in adf.columns:
                if col not in merged.columns:
                    merged[col] = adf[col]
                else:
                    ycol = pd.to_numeric(merged[col], errors="coerce")
                    acol = pd.to_numeric(adf[col], errors="coerce")
                    merged[col] = ycol.where(~ycol.isna(), acol)
        merged = merged.loc[:, ~pd.Index(merged.columns).duplicated(keep="last")]
        merged = _limit_years(merged, years)
        out[key_y] = merged
    return out

def compute_ttm(is_q: pd.DataFrame, cf_q: pd.DataFrame) -> Dict[str, float]:
    ttm = {}
    if is_q is not None and not is_q.empty:
        def _pick(row_names):
            for rn in row_names:
                if rn in is_q.index:
                    return pd.to_numeric(is_q.loc[rn], errors="coerce")
            return None
        rev = _pick(["Total Revenue", "TotalRevenue"])
        ni  = _pick(["Net Income", "NetIncome"])
        opi = _pick(["Operating Income", "OperatingIncome"])
        if rev is not None: ttm["Revenue_TTM"] = float(rev.iloc[-4:].sum())
        if ni  is not None: ttm["NetIncome_TTM"] = float(ni.iloc[-4:].sum())
        if opi is not None: ttm["OperatingIncome_TTM"] = float(opi.iloc[-4:].sum())

    if cf_q is not None and not cf_q.empty:
        def _pick_cf(row_names):
            for rn in row_names:
                if rn in cf_q.index:
                    return pd.to_numeric(cf_q.loc[rn], errors="coerce")
            return None
        ocf = _pick_cf(["Operating Cash Flow", "OperatingCashFlow"])
        capex = _pick_cf(["Capital Expenditure", "CapitalExpenditure"])
        if ocf is not None:
            ocf_4 = float(ocf.iloc[-4:].sum())
            ttm["OCF_TTM"] = ocf_4
            if capex is not None:
                ttm["FCF_TTM"] = ocf_4 + float(capex.iloc[-4:].sum())
    return ttm

def compute_ratios_annual(is_sel: pd.DataFrame, bs_sel: pd.DataFrame, cf_sel: pd.DataFrame) -> pd.DataFrame:
    def num(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")

    cols = sorted([c for c in set(is_sel.columns) | set(bs_sel.columns) | set(cf_sel.columns)], key=lambda x: int(x) if str(x).isdigit() else 0)
    out = pd.DataFrame(index=[
        "Gross Margin", "Operating Margin", "Net Margin",
        "Current Ratio", "Debt / Equity", "FCF Margin",
    ], columns=cols, dtype="float64")

    try:
        rev = num(is_sel.loc["Total Revenue"])
        gp  = num(is_sel.loc["Gross Profit"])
        opi = num(is_sel.loc["Operating Income"])
        ni  = num(is_sel.loc["Net Income"])
        ca  = num(bs_sel.loc["Total Current Assets"])
        cl  = num(bs_sel.loc["Total Current Liabilities"])
        debt = num(bs_sel.loc["Total Debt"])
        eq   = num(bs_sel.loc["Total Equity"])
        ocf  = num(cf_sel.loc["Operating Cash Flow"])
        capex = num(cf_sel.loc["Capital Expenditure"])
        fcf = ocf + capex

        out.loc["Gross Margin"] = gp / rev
        out.loc["Operating Margin"] = opi / rev
        out.loc["Net Margin"] = ni / rev
        out.loc["Current Ratio"] = ca / cl
        out.loc["Debt / Equity"] = debt / eq
        out.loc["FCF Margin"] = fcf / rev
    except Exception:
        pass
    return out

# ───────────────────────────── Price/Meta & Multiples ─────────────────────────────────

def fetch_price_meta(yf_symbol: str) -> Tuple[dict, dict]:
    tk = yf.Ticker(yf_symbol)
    fast = {}
    try:
        fi = tk.fast_info
        fast = {
            "last_price": float(fi["last_price"]),
            "currency": fi.get("currency"),
            "shares": float(fi.get("shares") or 0) if fi.get("shares") is not None else None,
            "market_cap": float(fi.get("market_cap") or 0) if fi.get("market_cap") is not None else None,
            "year_high": fi.get("year_high"),
            "year_low": fi.get("year_low"),
        }
    except Exception:
        # Fallback to history for last price
        try:
            px = tk.history(period="5d")["Close"].dropna()
            if not px.empty:
                fast["last_price"] = float(px.iloc[-1])
        except Exception:
            pass

    info = tk.info or {}
    # Normalize a few keys
    info_norm = {
        "shortName": info.get("shortName"),
        "longName": info.get("longName"),
        "longBusinessSummary": info.get("longBusinessSummary"),
        "website": info.get("website"),
        "exchange": info.get("exchange") or info.get("fullExchangeName"),
        "currency": info.get("currency") or fast.get("currency"),
        "country": info.get("country"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "fullTimeEmployees": info.get("fullTimeEmployees"),
        "fiscalYearEnd": info.get("fiscalYearEnd"),
        "city": info.get("city"),
        "state": info.get("state"),
        "zip": info.get("zip"),
        "address1": info.get("address1"),
        # market data
        "marketCap": info.get("marketCap") or fast.get("market_cap"),
        "enterpriseValue": info.get("enterpriseValue"),
        "sharesOutstanding": info.get("sharesOutstanding") or fast.get("shares"),
        "trailingEps": info.get("trailingEps"),
        "beta": info.get("beta"),
        "trailingAnnualDividendRate": info.get("trailingAnnualDividendRate"),
        "trailingAnnualDividendYield": info.get("trailingAnnualDividendYield"),
        "dividendYield": info.get("dividendYield"),
        "payoutRatio": info.get("payoutRatio"),
        "ebitda": info.get("ebitda"),   # often TTM
    }
    return fast, info_norm

def price_history_monthly(yf_symbol: str, years: int = 3) -> pd.DataFrame:
    tk = yf.Ticker(yf_symbol)
    df = tk.history(period=f"{years}y", interval="1mo")["Close"].dropna()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.to_frame("Close")
    df.index = df.index.strftime("%Y-%m")
    return df.T  # one-row, many months as columns

def build_multiples(yf_symbol: str, ttm: Dict[str, float], bs_sel: pd.DataFrame, info: dict, fast: dict) -> pd.DataFrame:
    price = fast.get("last_price")
    shares = info.get("sharesOutstanding") or fast.get("shares")
    market_cap = info.get("marketCap")
    if market_cap is None and (price is not None and shares):
        market_cap = float(price) * float(shares)

    # Cash & Total Debt from latest annual Selected BS
    cash = None; debt = None; book_equity = None
    if bs_sel is not None and not bs_sel.empty:
        latest_col = sorted([c for c in bs_sel.columns if str(c).isdigit()], key=lambda x: int(x))[-1] if len(bs_sel.columns) else None
        if latest_col:
            cash = pd.to_numeric(bs_sel.at["Cash & Equivalents", latest_col], errors="coerce")
            debt = pd.to_numeric(bs_sel.at["Total Debt", latest_col], errors="coerce")
            book_equity = pd.to_numeric(bs_sel.at["Total Equity", latest_col], errors="coerce")

    ev = info.get("enterpriseValue")
    if ev is None and market_cap is not None:
        net_debt = (float(debt) if pd.notna(debt) else 0) - (float(cash) if pd.notna(cash) else 0)
        ev = float(market_cap) + net_debt

    # TTM fundamentals
    rev_ttm = ttm.get("Revenue_TTM")
    ni_ttm  = ttm.get("NetIncome_TTM")
    ocf_ttm = ttm.get("OCF_TTM")
    fcf_ttm = ttm.get("FCF_TTM")
    opi_ttm = ttm.get("OperatingIncome_TTM")
    ebitda_ttm = info.get("ebitda")

    eps_ttm = None
    if shares and ni_ttm is not None:
        eps_ttm = float(ni_ttm) / float(shares)
    if eps_ttm is None:
        eps_ttm = info.get("trailingEps")

    # Multiples (float division with guards)
    def safe_div(a, b):
        try:
            if a is None or b in (None, 0): return None
            return float(a) / float(b) if float(b) != 0 else None
        except Exception:
            return None

    multiples = {
        "Price": price,
        "Market Cap": market_cap,
        "Enterprise Value": ev,
        "Shares Outstanding": shares,
        "Revenue (TTM)": rev_ttm,
        "Operating Income (TTM)": opi_ttm,
        "Net Income (TTM)": ni_ttm,
        "EBITDA (TTM)": ebitda_ttm,
        "OCF (TTM)": ocf_ttm,
        "FCF (TTM)": fcf_ttm,
        "Book Equity (latest)": float(book_equity) if book_equity is not None else None,
        "Net Debt (latest)": (float(debt or 0) - float(cash or 0)) if (debt is not None or cash is not None) else None,
        "Dividend Yield": info.get("dividendYield") or info.get("trailingAnnualDividendYield"),
        "Payout Ratio": info.get("payoutRatio"),
        # Valuation multiples
        "P/E (TTM)": safe_div(price, eps_ttm),
        "P/S (TTM)": safe_div(market_cap, rev_ttm),
        "P/B (latest)": safe_div(market_cap, book_equity),
        "P/CF (TTM)": safe_div(market_cap, ocf_ttm),
        "EV/Sales (TTM)": safe_div(ev, rev_ttm),
        "EV/EBITDA (TTM)": safe_div(ev, ebitda_ttm),
        "EV/FCF (TTM)": safe_div(ev, fcf_ttm),
    }
    df = pd.DataFrame(multiples, index=["Value"]).T
    return df

# ───────────────────────────── Streamlit Page ─────────────────────────────────────────

def run_fin_stat_downloader():
    load_dotenv()

    st.markdown("## 📥 Multi-Tab Financials (Yahoo + optional Alpha Vantage backfill)")
    st.caption("IS/BS/CF Annual & Quarterly, TTM, Ratios, Multiples, Metadata (with company background).")
    st.markdown("---")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        raw = st.text_input("Ticker (Yahoo symbol or RIC-style)", value="ADSK", placeholder="e.g., AAPL, ADSK, AIR.NZ, TSM")
    with c2:
        yrs = st.slider("Annual years (max 10)", 3, 10, 10)
    with c3:
        qn = st.slider("Quarterly periods", 4, 12, 8)

    if st.button("Fetch", type="primary"):
        if not raw.strip():
            st.error("Please enter a ticker.")
            return

        yf_symbol = _norm_ticker_for_yf(raw)
        base_symbol = yf_symbol.split(".")[0]    # for Alpha Vantage (US)
        av_key = os.getenv("ALPHAVANTAGE_API_KEY", "")

        try:
            # Yahoo statements
            with st.spinner(f"Fetching Yahoo statements for {yf_symbol}…"):
                y = fetch_yahoo_all(yf_symbol)

            # Selected + TTM
            with st.spinner("Building selected tables & TTM…"):
                sel = build_selected_tables(y)
                ttm = compute_ttm(y["IS_Q"], y["CF_Q"])

            # Backfill annuals with AV if key present
            with st.spinner("Backfilling older annuals (Alpha Vantage) if available…"):
                av = fetch_alpha_vantage_annual(base_symbol, av_key) if av_key else {"IS_A": pd.DataFrame(), "BS_A": pd.DataFrame(), "CF_A": pd.DataFrame()}
                merged = merge_annual_yahoo_alpha(y, av, yrs)

            # Limit quarterly to chosen qn
            IS_Q = _limit_quarters(y["IS_Q"], qn)
            BS_Q = _limit_quarters(y["BS_Q"], qn)
            CF_Q = _limit_quarters(y["CF_Q"], qn)

            # Selected (limit to yrs)
            IS_A_SEL = _limit_years(sel["IS_A_SEL"], yrs)
            BS_A_SEL = _limit_years(sel["BS_A_SEL"], yrs)
            CF_A_SEL = _limit_years(sel["CF_A_SEL"], yrs)

            # Ratios
            ratios = compute_ratios_annual(IS_A_SEL, BS_A_SEL, CF_A_SEL)

            # Price/Meta & Multiples
            fast, info = fetch_price_meta(yf_symbol)
            multiples = build_multiples(yf_symbol, ttm, BS_A_SEL, info, fast)
            price_hist = price_history_monthly(yf_symbol, years=3)

            # Metadata table (with long business summary)
            meta = pd.DataFrame.from_dict({
                "Symbol": [yf_symbol],
                "Short Name": [info.get("shortName")],
                "Long Name": [info.get("longName")],
                "Currency": [info.get("currency")],
                "Exchange": [info.get("exchange")],
                "Country": [info.get("country")],
                "Industry": [info.get("industry")],
                "Sector": [info.get("sector")],
                "Employees": [info.get("fullTimeEmployees")],
                "Fiscal Year End": [info.get("fiscalYearEnd")],
                "Website": [info.get("website")],
                "Address": [info.get("address1")],
                "City": [info.get("city")],
                "State": [info.get("state")],
                "ZIP": [info.get("zip")],
                "Beta": [info.get("beta")],
                "52W High": [fast.get("year_high")],
                "52W Low": [fast.get("year_low")],
                "Business Summary": [info.get("longBusinessSummary")],
            }).T
            meta.columns = ["Value"]

            # ── PREVIEW ─────────────────────────────────────────────────────────────
            st.subheader("Preview — Annual Selected (IS/BS/CF)")
            colA, colB, colC = st.columns(3)
            with colA: show_df(IS_A_SEL)
            with colB: show_df(BS_A_SEL)
            with colC: show_df(CF_A_SEL)

            st.subheader("Preview — Quarterly")
            col1, col2, col3 = st.columns(3)
            with col1: show_df(IS_Q)
            with col2: show_df(BS_Q)
            with col3: show_df(CF_Q)

            st.subheader("TTM Summary")
            if ttm:
                show_df(pd.DataFrame(ttm, index=["Value"]).T)
            else:
                st.write("TTM not available.")

            st.subheader("Multiples Snapshot")
            show_df(multiples)

            st.subheader("Ratios (Annual)")
            show_df(ratios)

            st.subheader("Metadata")
            show_df(meta)

            # ── EXCEL EXPORT ───────────────────────────────────────────────────────
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                # Annual Full (merged for deeper history)
                arrow_friendly(merged["IS_A"]).to_excel(writer, sheet_name="IS_Annual_Full")
                arrow_friendly(merged["BS_A"]).to_excel(writer, sheet_name="BS_Annual_Full")
                arrow_friendly(merged["CF_A"]).to_excel(writer, sheet_name="CF_Annual_Full")

                # Annual Selected
                arrow_friendly(IS_A_SEL).to_excel(writer, sheet_name="IS_Annual_Selected")
                arrow_friendly(BS_A_SEL).to_excel(writer, sheet_name="BS_Annual_Selected")
                arrow_friendly(CF_A_SEL).to_excel(writer, sheet_name="CF_Annual_Selected")

                # Quarterly (last periods)
                arrow_friendly(IS_Q).to_excel(writer, sheet_name=f"IS_Quarterly_{qn}q")
                arrow_friendly(BS_Q).to_excel(writer, sheet_name=f"BS_Quarterly_{qn}q")
                arrow_friendly(CF_Q).to_excel(writer, sheet_name=f"CF_Quarterly_{qn}q")

                # TTM, Ratios, Multiples
                if ttm:
                    arrow_friendly(pd.DataFrame(ttm, index=["Value"]).T).to_excel(writer, sheet_name="TTM")
                arrow_friendly(ratios).to_excel(writer, sheet_name="Ratios_Annual")
                arrow_friendly(multiples).to_excel(writer, sheet_name="Multiples_Snapshot")

                # 3y monthly price
                if not price_hist.empty:
                    arrow_friendly(price_hist).to_excel(writer, sheet_name="Price_History_Monthly")

                # Metadata (with long business summary)
                arrow_friendly(meta).to_excel(writer, sheet_name="Metadata")

            buf.seek(0)
            st.download_button(
                "⬇️ Download Excel",
                data=buf,
                file_name=f"{yf_symbol}_Financials_MultiTab.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        except Exception as e:
            st.error(f"Error: {type(e).__name__}: {e}")
