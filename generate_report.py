# generate_report.py

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF

# ─── PDF Report Generator ────────────────────────────────────────────────
def generate_financial_report(ticker: str,
                               income_df: pd.DataFrame,
                               cashflow_df: pd.DataFrame,
                               balance_df: pd.DataFrame,
                               projections: dict,
                               save_dir: Path):
    
    # Ensure output folder exists
    save_dir.mkdir(parents=True, exist_ok=True)

    # ─── 1. Generate Charts ───────────────────────────────────────────────
    charts = []

    def save_chart(data, title, ylabel, filename):
        plt.figure()
        data.plot(marker='o')
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel("Year")
        path = save_dir / filename
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        charts.append(path)

    # Chart 1: Revenue from Income Statement
    if "Revenue" in income_df.index:
        revenue_series = income_df.loc["Revenue"].dropna()
        save_chart(revenue_series,
                   f"{ticker} Revenue", "USD ($M)",
                   f"{ticker}_revenue.png")

    # Chart 2: FCFF Historical (if available)
    if "FCFF" in cashflow_df.index:
        fcff_series = cashflow_df.loc["FCFF"].dropna()
        save_chart(fcff_series,
                   f"{ticker} FCFF", "USD ($M)",
                   f"{ticker}_fcff.png")

    # ─── 2. Build PDF Report ──────────────────────────────────────────────
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"📄 Financial Report: {ticker}", ln=True)

    # Section: Historical Revenue
    if "Revenue" in income_df.index:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Revenue (Historical):", ln=True)
        revenue_path = save_dir / f"{ticker}_revenue.png"
        if revenue_path.exists():
            pdf.image(str(revenue_path), w=180)

    # Section: Historical FCFF
    if "FCFF" in cashflow_df.index:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "FCFF (Historical):", ln=True)
        fcff_path = save_dir / f"{ticker}_fcff.png"
        if fcff_path.exists():
            pdf.image(str(fcff_path), w=180)

    # Section: LLM Projections
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "LLM-Based Valuation Projections:", ln=True)
    pdf.set_font("Arial", "", 12)
    for key, value in projections.items():
        pdf.cell(0, 10, f"{key}: {value}", ln=True)

    # ─── 3. Save PDF File ─────────────────────────────────────────────────
    pdf_output_path = save_dir / f"{ticker}_financial_report.pdf"
    pdf.output(str(pdf_output_path))
