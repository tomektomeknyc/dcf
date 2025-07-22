# fin_report_tab.py
import streamlit as st
from pathlib import Path
from generate_report import generate_financial_report
from historical_data_extractor import extract_all_sheets


def render_fin_report_tab(selected_stocks):
    st.markdown("## 📄 Financial Report Generator")

    selected_stocks = st.session_state.get("selected_stocks", [])
    if not selected_stocks:
        st.warning("No stocks selected. Please select stocks on the main dashboard first.")
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

            # STEP 2: LLM-based simulated projections (placeholder)
            projections = {
                "r": 0.10,
                "g": 0.03,
                "FCFF2026": 2100,
                "intrinsic": 18.6
            }

            # STEP 3: Generate PDF
            try:
                save_dir = Path("reports") / ticker
                generate_financial_report(
                    ticker=ticker,
                    income_df=income_df,
                    cashflow_df=cashflow_df,
                    balance_df=balance_df,
                    projections=projections,
                    save_dir=save_dir
                )
                st.success(f"✅ Report for {ticker} generated.")

                with open(save_dir / f"{ticker}_financial_report.pdf", "rb") as f:
                    st.download_button(
                        label=f"📥 Download {ticker} PDF Report",
                        data=f,
                        file_name=f"{ticker}_financial_report.pdf",
                        mime="application/pdf"
                    )

            except Exception as e:
                st.error(f"❌ Failed to generate report for {ticker}: {e}")
