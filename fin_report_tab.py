# fin_report_tab.py
import streamlit as st
from pathlib import Path
from generate_report import generate_financial_report
from historical_data_extractor import extract_all_sheets
from fcf_calculations import compute_fcff


def render_fin_report_tab(selected_stocks=None):
    st.markdown("## 📄 Financial Report Generator")

    # Get selected stocks from session or as argument
    selected_stocks = st.session_state.get("selected_stocks", [])

    if not selected_stocks:
        st.warning("🚧 Under construction ")
        return

    st.markdown("### 📌 Select which stocks to generate reports for:")

    report_stocks = st.multiselect(
        label="Choose stocks to include in the report",
        options=selected_stocks,
        default=selected_stocks,
        key="report_stocks"
    )

    if not report_stocks:
        st.info("No stocks selected for report generation.")
        return

    if st.button("📝 Generate Financial Report"):
        for ticker in report_stocks:
            st.info(f"Generating report for {ticker}...")

            # STEP 1: Extract historical data
            try:
                sheets = extract_all_sheets(ticker)
                income_df = sheets.get("Income Statement")
                cashflow_df = sheets.get("Cash Flow")
                balance_df = sheets.get("Balance Sheet")
            except Exception as e:
                st.error(f"❌ Failed to load data for {ticker}: {e}")
                continue
            
            # After the try/except block, inside the for ticker in report_stocks loop:
            fcff_series = compute_fcff(
                ticker=ticker,
                df_is=income_df,
                df_cf=cashflow_df,
                df_bs=balance_df,
                year_col="Year"
            )

            try:
                # Use your projected years here if you want all 5, or just pick the last for FCFF2026
                fcff_proj = [fcff_series.iloc[-5], fcff_series.iloc[-4], fcff_series.iloc[-3], fcff_series.iloc[-2], fcff_series.iloc[-1]]
            except Exception:
                fcff_proj = [None] * 5

            # Terminal value logic (already done in compute_fcff, but just in case)
            try:
                terminal_value = fcff_series.loc["Terminal Value"] if "Terminal Value" in fcff_series.index else None
            except Exception:
                terminal_value = None

            # Build the projections dict
            projections = {
                "FCFF_projections": fcff_proj,
                "FCFF2026": fcff_proj[-1] if fcff_proj else None,
                "terminal_value": terminal_value,
                "g": 0.03,  # Replace with your actual g if you have
                "r": 0.10,  # Replace with your actual r if you have
                "intrinsic": 29.91,  # Example, replace with your computed intrinsic value per share
            }

            # STEP 2: LLM-based simulated projections (placeholder)
            # After FCFF series is generated, extract projected values for report
            projections = {
                "g": 0.03,  # Replace with actual value if you have it!
                "r": 0.10,  # Replace with actual value if you have it!
                "FCFF2026": None,
                "intrinsic": None,
            }

            # Extract FCFF2026 if it exists in the series (use your projected year)
            try:
                projections["FCFF2026"] = fcff_series.loc[2026] if 2026 in fcff_series.index else None
            except Exception:
                projections["FCFF2026"] = None

           
            # For now, placeholder logic (to be replaced with  LLM/model results):
            projections["g"] = 0.03   # or however I compute "g"
            projections["r"] = 0.10   # or however I  compute "r"
            projections["intrinsic"] = 18.6  # placeholder, will replace in next step


            # STEP 3: Generate HTML
            try:
                save_dir = Path("reports") / ticker
                html_path = generate_financial_report(
                    ticker=ticker,
                    income_df=income_df,
                    cashflow_df=cashflow_df,
                    balance_df=balance_df,
                    projections=projections,
                    save_dir=save_dir
                )
                st.success(f"✅ Report for {ticker} generated.")

                with open(html_path, "r", encoding="utf-8") as f:
                    st.download_button(
                        label=f"📥 Download {ticker} HTML Report",
                        data=f,
                        file_name=f"{ticker}_financial_report.html",
                        mime="text/html"
                    )

            except Exception as e:
                st.error(f"❌ Failed to generate report for {ticker}: {e}")
