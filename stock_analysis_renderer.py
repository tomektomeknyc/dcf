# stock_analysis_renderer.py
import streamlit as st
import matplotlib.pyplot as plt
import os
from pathlib import Path

def render_financial_analysis(selected_stocks):
    """
    Renders financial analysis for each selected stock
    """
    for stock in selected_stocks:
        st.subheader(f"📊 Financial Analysis: {stock}")

        # Load pre-generated charts if available (e.g. FCFF, Revenue)
        image_dir = Path("generated_reports")

        chart_files = [
            image_dir / f"{stock}_revenue.png",
            image_dir / f"{stock}_fcff.png"
        ]

        charts_loaded = False
        for chart_path in chart_files:
            if chart_path.exists():
                st.image(str(chart_path), caption=chart_path.name)
                charts_loaded = True

        if not charts_loaded:
            st.info("No charts available yet. Run the computation/report tab to generate visuals.")

        # Example KPI summary placeholder
        st.markdown("""
        - ✅ Revenue CAGR: *Placeholder*
        - 💰 Average FCFF: *Placeholder*
        - 📉 Valuation Discount: *Placeholder*
        """)
