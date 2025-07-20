# regression_engine.py
import pandas as pd
import statsmodels.api as sm



def compute_capm_beta(stock_returns: pd.Series, ff5_data: pd.DataFrame) -> dict:
    aligned = ff5_data.copy()
    aligned = aligned.loc[stock_returns.index]
    excess_stock = stock_returns - aligned["RF"]

    X = aligned[["Mkt-RF"]]
    y = excess_stock

    model = sm.OLS(y, sm.add_constant(X)).fit()

    return {
        "market_beta": float(model.params["Mkt-RF"]),
        "alpha": float(model.params["const"]),
        "r_squared": float(model.rsquared),
        "residuals": model.resid.tolist(),
        "dates": aligned.index.strftime("%Y-%m").tolist()
    }


def compute_ff5_betas(stock_returns: pd.Series, ff5_data: pd.DataFrame) -> dict:
    # 1) Ensure all factor columns are numeric
    factors = ff5_data[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]].apply(
        pd.to_numeric, errors="coerce"
    )

    # 2) Align on common dates
    common_idx = stock_returns.index.intersection(factors.index)
    stock_ret_aligned = stock_returns.loc[common_idx].astype(float)
    factors_aligned = factors.loc[common_idx]

    # 3) Compute excess returns
    excess_stock = stock_ret_aligned - factors_aligned["RF"]

    # 4) Run the multivariate regression
    X = factors_aligned[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]
    y = excess_stock
    model = sm.OLS(y, sm.add_constant(X)).fit()

    return {
        "market_beta": float(model.params["Mkt-RF"]),
        "smb_beta":    float(model.params["SMB"]),
        "hml_beta":    float(model.params["HML"]),
        "rmw_beta":    float(model.params["RMW"]),
        "cma_beta":    float(model.params["CMA"]),
        "alpha":       float(model.params["const"]),
        "r_squared":   float(model.rsquared),
        "residuals":   model.resid.tolist(),
        "dates":       factors_aligned.index.strftime("%Y-%m").tolist(),
    }
# In regression_engine.py, alongside compute_ff5_betas and compute_ff5_betas:


def compute_ff5_residuals(stock_ret: pd.Series, factor_ret: pd.DataFrame) -> pd.Series:
    """
    Run an OLS of stock_ret on the five FF‐5 factors in factor_ret,
    return the residual series (actual minus predicted).
    """
    # 1) Align on dates & drop any missing
    df = pd.concat([stock_ret.rename("y"), factor_ret], axis=1).dropna()
    # 2) Build X matrix with constant
    X = sm.add_constant(df[factor_ret.columns])
    # 3) Fit OLS and return residuals
    model = sm.OLS(df["y"], X).fit()
    return model.resid

def compute_industry_residuals(
    stock_ret: pd.Series,
    market_ret: pd.Series,
    beta: float,
    risk_free_rate: float
) -> pd.Series:
    """
    αₜ = actual stock return − [RF + β·(market return − RF)]
    """
    # Align dates
    df = pd.concat([stock_ret.rename("R"), market_ret.rename("M")], axis=1).dropna()
    # Excess market over RF
    df["M_ex"] = df["M"] - risk_free_rate
    # Predicted = RF + β·M_ex
    df["pred"] = risk_free_rate + beta * df["M_ex"]
    # Residual = actual − predicted
    return (df["R"] - df["pred"]).rename("alpha")
