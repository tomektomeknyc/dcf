#generate_report.py
import os

def generate_financial_report(stock, income_df=None, cashflow_df=None, balance_df=None, projections=None, save_dir="reports_demo"):
    os.makedirs(save_dir, exist_ok=True)
    html = f"""
    <html>
    <head>
        <title>{stock['Ticker']} Financial Report</title>
        <style>
            body {{ font-family: Arial; background: #181C20; color: #FAFAFA; padding: 2em; }}
            h1 {{ color: #75FF43; }}
            table {{ border-collapse: collapse; width: 50%; }}
            th, td {{ border: 1px solid #333; padding: 8px; text-align: center; }}
            th {{ background: #232931; color: #75FF43; }}
        </style>
    </head>
    <body>
        <h1>{stock['Ticker']} – Financial Metrics</h1>
        <table>
            <tr><th>CAPM Beta</th><td>{stock.get('CAPM_Beta','')}</td></tr>
            <tr><th>FF5 Beta</th><td>{stock.get('FF5_Beta','')}</td></tr>
            <tr><th>Damodaran Beta</th><td>{stock.get('Damodaran_Beta','')}</td></tr>
            <tr><th>Intrinsic Value</th><td>{projections.get('intrinsic','')}</td></tr>
            <tr><th>Discount Rate (r)</th><td>{projections.get('r','')}</td></tr>
            <tr><th>Growth Rate (g)</th><td>{projections.get('g','')}</td></tr>
            <tr><th>FCFF 2026 (simulated)</th><td>{projections.get('FCFF2026','')}</td></tr>
        </table>
    </body>
    </html>
    """
    out_path = os.path.join(save_dir, f"{stock['Ticker']}_financial_report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved {out_path}")
    return out_path


