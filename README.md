DCF Valuation Application with LLM-Enhanced Projections
This financial analysis platform combines traditional DCF (Discounted Cash Flow) methodology with artificial intelligence to provide comprehensive stock valuations using multiple WACC calculation approaches.

📈 Core Methodology
1. Free Cash Flow to Firm (FCFF) Calculation:

FCFF = EBIT × (1 - Tax Rate) + Depreciation - CapEx - Δ NWC

Where:

EBIT = Earnings Before Interest and Taxes
Tax Rate = Income Taxes Paid / Pre-tax Income
Δ NWC = Change in Net Working Capital
2. DCF Valuation Framework (5 Key Formulas):

Formula 1 - Present Value of Forecast FCFF:

PV(FCFF) = Σ[FCFF_t / (1 + WACC)^t] for t = 1 to 5

Formula 2 - Present Value of Terminal Value:

PV(Terminal Value) = [FCFF_5 × (1 + g) / (WACC - g)] / (1 + WACC)^5

Where g = long-term growth rate

Formula 3 - Enterprise Value:

Enterprise Value = PV(FCFF) + PV(Terminal Value)

Formula 4 - Equity Value:

Equity Value = Enterprise Value - Net Debt
Net Debt = Total Debt - Cash and Cash Equivalents

Formula 5 - Per-Share Intrinsic Value:

Intrinsic Value per Share = Equity Value / Shares Outstanding

🧠 AI-Enhanced Projections
LLM Integration: Uses GPT-4o for intelligent FCFF projections that consider:

Industry-specific growth patterns
Economic conditions and market cycles
Company fundamentals and competitive positioning
Historical performance trends
Traditional vs AI Approach:

Traditional: Simple growth rate assumptions
My Method: Contextual analysis with reasoning for each projection year LLM‑Driven Rate & Cash‑Flow Forecasting: I leveraged a fine‑tuned GPT‑3.5‑turbo model to generate unbiased, data‑grounded assumptions for long‑term terminal growth rate (g) and discount rate (r). By feeding in each company’s trailing 5‑year EBIT CAGR, average tax rate, D&A growth, CapEx growth, and ΔNWC/Sales, the LLM contextualizes these metrics against industry‑ and region‑specific norms—avoiding manual anchoring bias. It also projects the next fiscal year’s FCFF by applying those inferred rates directly to the most recent financials, ensuring our one‑year forecast combines rigorous historical analysis with AI‑powered insight.
📊 WACC Calculation Methods
1. CAPM (Capital Asset Pricing Model):

WACC = (E/V × Re) + (D/V × Rd × (1-T))
Re = Rf + β × (Rm - Rf)

2. Fama-French 5-Factor Model:

R - Rf = α + β(Rm-Rf) + sSMB + hHML + rRMW + cCMA + ε

Factors: Market, Size, Value, Profitability, Investment

3. Damodaran Industry Beta Method:

Uses industry-specific beta coefficients
Regional adjustments for market conditions
Leveraged/unleveraged beta calculations
🎯 Key Features
Multiple Valuation Perspectives: Three independent WACC calculations
Real Financial Data: Authentic company financials and market data
Interactive Visualization: Dynamic charts and comparative analysis
Professional Methodology: Industry-standard DCF implementation
AI-Powered Insights: Intelligent cash flow projections
📋 Data Sources: Refinitiv

Company Financials: Income statements, balance sheets, cash flow statements
Market Data: Regional stock indices and risk-free rates
Beta Coefficients: Damodaran online data, Fama-French factors
Industry Classifications: Sector-specific analysis and benchmarking
🔍 Analysis Workflow
Select Companies from available dataset
Choose all Estimation Methods (CAPM, FF-5, Damodaran)
Run Calculations for beta coefficients and risk factors
Generate FCFF Projections using AI analysis
View DCF Results with intrinsic valuations
Compare Methods through interactive visualizations
