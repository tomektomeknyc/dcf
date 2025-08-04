#fin_report_tab.py
from pathlib import Path
import streamlit as st
from generate_report import generate_financial_report
from historical_data_extractor import extract_all_sheets
from fcf_calculations import compute_fcff
import matplotlib.pyplot as plt
from string import Template
import pickle

def render_fin_report_tab(selected_stocks=None):
    st.markdown("## 📊 Financial Report Generator")

    # 🔁 Load from session or pickle
    if selected_stocks is None:
        selected_stocks = st.session_state.get("selected_stocks", [])
        if not selected_stocks:
            try:
                with open("selected_stocks.pkl", "rb") as f:
                    selected_stocks = pickle.load(f)
            except Exception:
                selected_stocks = []

    # 🔁 Save state to ensure later use
    
    st.session_state["selected_stocks"] = selected_stocks
    with open("selected_stocks.pkl", "wb") as f:
        pickle.dump(selected_stocks, f)

    # 📌 Stock Picker
    st.markdown("### 📌 Select which stocks to generate reports for:")
    report_stocks = st.multiselect(
        label="Choose stocks to include in the report",
        options=selected_stocks,
        default=selected_stocks,
        key="report_stocks"
    )

    if not report_stocks:
        st.info("📎 No stocks selected for report generation.")
        return

    # 🧾 Report Generation
    if st.button("🧾 Generate Financial Report"):
        for ticker in report_stocks:
            st.info(f"🛠️ Generating report for {ticker}...")

            try:
                # Step 1: Extract historical data
                sheets = extract_all_sheets(ticker)
                income_df = sheets.get("Income Statement")
                cashflow_df = sheets.get("Cash Flow")
                balance_df = sheets.get("Balance Sheet")

                if not income_df or not cashflow_df or not balance_df:
                    raise ValueError("Missing one or more financial statements.")

                # Step 2: Compute FCFF
                fcff_series = compute_fcff(
                    ticker=ticker,
                    df_is=income_df,
                    df_cf=cashflow_df,
                    df_bs=balance_df,
                    year_col="Year"
                )

                # Step 3: Projection
                try:
                    fcff_proj = [
                        fcff_series.iloc[-5],
                        fcff_series.iloc[-4],
                        fcff_series.iloc[-3],
                        fcff_series.iloc[-2],
                        fcff_series.iloc[-1],
                    ]
                except Exception:
                    fcff_proj = [None] * 5

                try:
                    terminal_value = fcff_series.loc["Terminal Value"] if "Terminal Value" in fcff_series.index else None
                except Exception:
                    terminal_value = None

                try:
                    fcf2026 = fcff_series.loc[2026] if 2026 in fcff_series.index else None
                except Exception:
                    fcf2026 = None

                projections = {
                    "FCFF_projections": fcff_proj,
                    "FCFF2026": fcf2026,
                    "terminal_value": terminal_value,
                    "g": 0.03,       # Example growth rate
                    "r": 0.10,       # Example discount rate
                    "intrinsic": 18.6
                }

                # Step 3.5: Save chart
                chart_path = Path("reports") / ticker / "fcff_chart.png"
                try:
                    plt.figure(figsize=(8, 4))
                    plt.plot(range(2022, 2027), fcff_proj, marker='o', linestyle='--', color='blue')
                    plt.title(f"{ticker} FCFF Projections (2022–2026)")
                    plt.xlabel("Year")
                    plt.ylabel("FCFF ($M)")
                    plt.grid(True)
                    plt.tight_layout()
                    chart_path.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(chart_path)
                    plt.close()
                except Exception as e:
                    st.warning(f"⚠️ Failed to generate chart for {ticker}: {e}")
                    chart_path = ""

                # Step 4: Generate HTML Report
                save_dir = Path("reports") / ticker
                save_dir.mkdir(parents=True, exist_ok=True)

                try:
                    with open("templates/report_template.html", "r", encoding="utf-8") as f:
                        template_str = f.read()
                    html_template = Template(template_str)

                    html_filled = html_template.safe_substitute({
                        "ticker": ticker,
                        "wacc": projections.get("g", "N/A"),
                        "r": projections.get("r", "N/A"),
                        "tv": terminal_value if terminal_value else "N/A",
                        "fcff": projections.get("FCFF2026", "N/A"),
                        "chart_path": chart_path.as_posix(),
                        "llm_summary": "Placeholder for future LLM summary."
                    })

                    html_path = save_dir / f"{ticker}_report.html"
                    with open(html_path, "w", encoding="utf-8") as f:
                        f.write(html_filled)

                    import streamlit.components.v1 as components
                    with open(html_path, "r", encoding="utf-8") as report_file:
                        report_html = report_file.read()
                    components.html(report_html, height=1000, scrolling=True)

                    st.success(f"✅ Report for {ticker} generated.")
                    with open(html_path, "r", encoding="utf-8") as f:
                        st.download_button(
                            label=f"⬇️ Download {ticker} HTML Report",
                            data=f,
                            file_name=f"{ticker}_financial_report.html",
                            mime="text/html"
                        )

                except Exception as e:
                    st.error(f"❌ Failed to generate report for {ticker}: {e}")

            except Exception as e:
                st.error(f"❌ Failed to process {ticker}: {e}")
