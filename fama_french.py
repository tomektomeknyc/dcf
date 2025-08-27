# fama_french.py
"""
Auto-fetch Fama-French 5 factors, compute alphas & cache results.
"""
from __future__ import annotations
import time
import pandas as pd
import streamlit as st
from typing import Dict
from scrape_ff5 import get_ff5_data_by_folder
#from ticker_to_industry import ticker_to_region

from typing import List, Dict, Set
import time

from typing import Dict

from ticker_to_industry import ticker_industry_map

from ticker_to_region import ticker_to_region


##### FAMA FRENCH START #####


def run_fama_french(
    sel_tickers: List[str],
    methods: Set[str] | None = None,
) -> tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Fetch Fama–French 5 factors per selected ticker and return:
    (ff5_results, ff5_dict, ff5_df_concat)

    Safe to call on every rerun. If FF-5 is not in `methods` or no tickers,
    returns ({}, {}, empty DataFrame).
    """
    if not sel_tickers or (methods is not None and "FF-5" not in methods):
        return {}, {}, pd.DataFrame()

    # progress/marker in UI
    st.session_state["ff5_ran"] = True
    ph_ff5 = st.sidebar.empty()
    ph_ff5.markdown("### 🔄 Auto-Updating FF-5 Factors")

    ff5_results: Dict[str, pd.DataFrame] = {}
    for ticker in sel_tickers:
        folder = ticker_to_region(ticker)
        try:
            df_ff5 = get_ff5_data_by_folder(ticker, folder)
            ff5_results[ticker] = df_ff5
        except Exception as e:
            st.sidebar.error(f"✗ {ticker}: {e}")

    if ff5_results:
        st.session_state["ff5"] = ff5_results
        st.session_state["ff5_ran"] = True
        time.sleep(0.2)
        ph_ff5.empty()
        ff5_df_concat = pd.concat(ff5_results.values(), axis=1)
    else:
        st.session_state.pop("ff5", None)
        st.session_state["ff5_ran"] = False
        ff5_df_concat = pd.DataFrame()

    # Return both the dict and the concatenated wide DF
    return ff5_results, ff5_results, ff5_df_concat
##### FAMA FRENCH END #####

__all__ = ["run_fama_french"]
