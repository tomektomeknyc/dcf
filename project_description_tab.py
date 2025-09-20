import streamlit as st

def render_project_description_tab():
    """
    Renders the Project Description tab content
    """
    st.markdown('<div class="project-description">', unsafe_allow_html=True)
    st.markdown("""
    ## 🚀 DCF Valuation Application with LLM-Enhanced Projections
    
    This financial analysis platform combines traditional DCF (Discounted Cash Flow) methodology with artificial intelligence to provide comprehensive stock valuations using multiple WACC calculation approaches.
    
    ### 📈 Core Methodology
    
    **1. Free Cash Flow to Firm (FCFF) Calculation:**
    ```
    FCFF = EBIT × (1 - Tax Rate) + Depreciation - CapEx - Δ NWC
    ```
    Where:
    - EBIT = Earnings Before Interest and Taxes
    - Tax Rate = Income Taxes Paid / Pre-tax Income
    - Δ NWC = Change in Net Working Capital
    
    **2. DCF Valuation Framework (5 Key Formulas):**
    
    **Formula 1 - Present Value of Forecast FCFF:**
    ```
    PV(FCFF) = Σ[FCFF_t / (1 + WACC)^t] for t = 1 to 5
    ```
    
    **Formula 2 - Present Value of Terminal Value:**
    ```
    PV(Terminal Value) = [FCFF_5 × (1 + g) / (WACC - g)] / (1 + WACC)^5
    ```
    Where g = long-term growth rate
    
    **Formula 3 - Enterprise Value:**
    ```
    Enterprise Value = PV(FCFF) + PV(Terminal Value)
    ```
    
    **Formula 4 - Equity Value:**
    ```
    Equity Value = Enterprise Value - Net Debt
    Net Debt = Total Debt - Cash and Cash Equivalents
    ```
    
    **Formula 5 - Per-Share Intrinsic Value:**
    ```
    Intrinsic Value per Share = Equity Value / Shares Outstanding
    ```
    
    ### 🧠 AI-Enhanced Projections
    
    **LLM Integration:** Uses GPT-4o for intelligent FCFF projections that consider:
    - Industry-specific growth patterns
    - Economic conditions and market cycles
    - Company fundamentals and competitive positioning
    - Historical performance trends
    
    **Traditional vs AI Approach:**
    - Traditional: Simple growth rate assumptions
    - My Method: Contextual analysis with reasoning for each projection year
    LLM‑Driven Rate & Cash‑Flow Forecasting:
    I leveraged a fine‑tuned GPT‑3.5‑turbo model to generate unbiased, data‑grounded assumptions for long‑term 
    terminal growth rate (g) and discount rate (r). By feeding in each company’s trailing 5‑year EBIT CAGR, average 
    tax rate, D&A growth, CapEx growth, and ΔNWC/Sales, the LLM contextualizes these metrics against industry‑ and 
    region‑specific norms—avoiding manual anchoring bias. It also projects the next fiscal year’s FCFF by applying 
    those inferred rates directly to the most recent financials, ensuring our one‑year forecast combines rigorous 
    historical analysis with AI‑powered insight.
    ### 📊 WACC Calculation Methods
    
    **1. CAPM (Capital Asset Pricing Model):**
    ```
    WACC = (E/V × Re) + (D/V × Rd × (1-T))
    Re = Rf + β × (Rm - Rf)
    ```
    
    **2. Fama-French 5-Factor Model:**
    ```
    R - Rf = α + β(Rm-Rf) + sSMB + hHML + rRMW + cCMA + ε
    ```
    Factors: Market, Size, Value, Profitability, Investment
    
    **3. Damodaran Industry Beta Method:**
    - Uses industry-specific beta coefficients
    - Regional adjustments for market conditions
    - Leveraged/unleveraged beta calculations
    
    ### 🎯 Key Features
    
    - **Multiple Valuation Perspectives:** Three independent WACC calculations
    - **Real Financial Data:** Authentic company financials and market data
    - **Interactive Visualization:** Dynamic charts and comparative analysis
    - **Professional Methodology:** Industry-standard DCF implementation
    - **AI-Powered Insights:** Intelligent cash flow projections
    
    ### 📋 Data Sources
    Refinitiv:
    - **Company Financials:** Income statements, balance sheets, cash flow statements
    - **Market Data:** Regional stock indices and risk-free rates
    - **Beta Coefficients:** Damodaran online data, Fama-French factors
    - **Industry Classifications:** Sector-specific analysis and benchmarking
    
    ### 🔍 Analysis Workflow
    
    1. **Select Companies** from available dataset
    2. **Choose all Estimation Methods** (CAPM, FF-5, Damodaran)
    3. **Run Calculations** for beta coefficients and risk factors
    4. **Generate FCFF Projections** using AI analysis
    5. **View DCF Results** with intrinsic valuations
    6. **Compare Methods** through interactive visualizations
    
    ### 📐 Massey University Investment Club Formula References
    
    Below are the foundational formulas from our Massey University Investment Club sessions, while preparing for
    CFA Institute Research Challenge in Auckland 2024, providing the mathematical framework for DCF valuation:
    """)
    
    # FCFF and FCFE Formulas
    st.markdown("#### Free Cash Flow Formulas")
    st.image("images/FCFF_FCFE.jpg", caption="FCFF & FCFE Calculation Methods", width=300)
    with st.expander("Click to view full size"):
       st.image("images/FCFF_FCFE.jpg", caption="FCFF & FCFE Calculation Methods - Full Size")
    
    st.markdown("---")
    
    # Present Value of FCF
    st.markdown("#### Present Value of Free Cash Flow")
    st.image("images/PV_FCF.jpg", caption="Present Value DCF Calculations", width=300)
    with st.expander("Click to view full size"):
       st.image("images/PV_FCF.jpg", caption="Present Value DCF Calculations - Full Size")
    
    st.markdown("---")
    
    # Additional FCFE and FCFF Reference
    st.markdown("#### Extended FCFE & FCFF Analysis")
    st.image("images/FCFE_FCFF_2.jpg", caption="Advanced FCFE and FCFF Methods", width=300)
    with st.expander("Click to view full size"):
       st.image("images/FCFE_FCFF_2.jpg", caption="Advanced FCFE and FCFF Methods")
    
    st.markdown("---")
    
    # Valuation Methods Overview
    st.markdown("#### Valuation Methodologies")
    st.image("images/Valuation_Methods.jpg", caption="Complete Valuation Framework", width=300)
    with st.expander("Click to view full size"):
       st.image("images/Valuation_Methods.jpg", caption="Complete Valuation Framework")
    
    st.markdown("""
    ### 💡 Investment Insights
   
    The application helps identify:
    - **Undervalued stocks** (intrinsic value > market price)
    - **Overvalued securities** (intrinsic value < market price)
    - **Valuation consensus** across different methodologies
    - **Risk-adjusted returns** through comprehensive analysis
    
    *Note: This tool is for educational and analytical purposes. Investment decisions should consider multiple factors beyond DCF analysis.*
    """)
