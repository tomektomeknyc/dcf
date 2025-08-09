# damodaran.py

##### DAMODARAN START #####
"""
Load and process Damodaran industry betas for selected tickers.
"""
from pathlib import Path
import streamlit as st
from typing import Dict, List, Optional
#from capm import _download_region_beta, fetch_damodaran_industry_betas, find_industry_beta
from fetch_damodaran_betas import _download_region_beta, fetch_damodaran_industry_betas, find_industry_beta
# from ticker_to_industry import ticker_industry_map, ticker_to_region
from ticker_to_industry import ticker_industry_map
from ticker_to_region  import ticker_to_region



def load_damodaran_betas(sel_tickers: List[str]) -> Dict[str, Optional[float]]:
    """
    Pre-load and cache Damodaran betas for each region,
    then fetch industry betas for each ticker.
    Returns a dict mapping ticker -> beta or None.
    """
    # — Pre-load Damodaran betas for each region, but cache locally —
    damo_files: Dict[str, str] = {}
    for reg in ("US", "Europe", "AU_NZ"):  # extend as needed
        local_path = Path("Damodaran") / f"totalbeta{reg}.xls"
        if local_path.exists():
            damo_files[reg] = str(local_path)
        else:
            damo_files[reg] = _download_region_beta(reg)
    st.session_state["damo_files"] = damo_files

    # Fetch & cache the industry-beta table for the first ticker's region
    if sel_tickers:
        first_region = ticker_to_region(sel_tickers[0])
        st.session_state["damo_industry_df"] = fetch_damodaran_industry_betas(first_region)

    # Build beta map for each selected ticker
    damo_betas: Dict[str, Optional[float]] = {}
    for t in sel_tickers:
        region = ticker_to_region(t)
        df_damo = st.session_state.get("damo_industry_df")
        industry = ticker_industry_map.get(t)
        match = (find_industry_beta(df_damo, None, industry)
                 if industry and df_damo is not None
                 else None)
        damo_betas[t] = match["beta"] if match else None

    return damo_betas


def render_damodaran_sidebar(damo_betas: Dict[str, Optional[float]]):
    """
    Display toggle and Beta output in the Streamlit sidebar.
    """
    # ─── Sidebar toggle for Damodaran β in the combined chart ─────────────
    show_damo = st.sidebar.checkbox(
        "Show Damodaran β in the combined β-chart",
        value=False,
        key="show_damo_chart1",
    )

    # ─── Damodaran Industry Betas (sidebar) ───────────────────────────────
    if damo_betas:
        st.sidebar.markdown("### 📟 Damodaran Industry Betas")
        beta_slot = st.sidebar.empty()
        last_tkr, last_beta = next(reversed(damo_betas.items()))
        if last_beta is not None:
            beta_slot.success(f"{last_tkr} β = {last_beta:.2f}")
        else:
            beta_slot.write(f"{last_tkr} β = n/a")



__all__ = ["load_damodaran_betas", "render_damodaran_sidebar"]

##### DAMODARAN END #####
