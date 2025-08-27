# hedge_prompt.py
# Builds an investor-facing LLM prompt from HedgeStats.

from __future__ import annotations
from dataclasses import asdict
from typing import Optional
from risk_hedge import HedgeStats

def build_hedge_investor_prompt(
    stats: HedgeStats,
    *,
    asset_name: str = "Spot Asset",
    futures_name: str = "Futures / Proxy",
    return_freq: str = "daily returns",
    sample_window: str = "(specify window)",
    notional_amount: float = 1_000_000.0,
    unit_label: str = "contracts",
    fut_multiplier: float = 1.0,   # e.g., each futures contract hedges N units of spot
) -> str:
    # Safe guards
    var_reduction_pct = 0.0
    if stats.var_unhedged > 0:
        var_reduction_pct = 100.0 * (1.0 - (stats.var_min / stats.var_unhedged))

    # Example sizing per 1 unit of spot exposure (scale in your UI by notional)
    h_star_scaled = stats.h_star / (fut_multiplier if fut_multiplier else 1.0)

    # Keep the prompt crisp, specific, investor-friendly
    prompt = f"""
You are a buy-side analyst. Explain the hedge analysis below to a non-technical investor.
Keep it concise, structured, and decision-oriented. Avoid jargon; define any unavoidable term in one short line.

--- INPUT DATA ---
Asset (spot): {asset_name}
Hedge instrument (futures/proxy): {futures_name}
Return frequency: {return_freq}
Sample window: {sample_window}

Spot volatility (σ_S): {stats.sigma_S:.4f}
Futures volatility (σ_F): {stats.sigma_F:.4f}
Correlation (ρ): {stats.rho:.2f}

Optimal hedge ratio (h*): {stats.h_star:.3f}
Unhedged variance: {stats.var_unhedged:.6f}
Minimum variance at h*: {stats.var_min:.6f}
Variance reduction: {var_reduction_pct:.1f}%

--- WHAT TO DO ---
1) Executive Summary (2–3 bullets): say what h* means in plain English and the degree of risk reduction.
2) Explain each number briefly:
   - σ_S, σ_F: volatility (typical size of moves)
   - ρ: how similarly the two move (negative means they offset)
   - h*: how many hedge units per 1 unit of spot (direction if ρ<0)
3) Practical hedge plan:
   - For a notional of {notional_amount:,.0f}, suggest ~{h_star_scaled:.3f} {unit_label} per 1 unit of spot exposure (scale to notional and contract multiplier).
   - Clarify this is variance (risk) minimization, not guaranteed P&L improvement.
4) Sensitivities / caveats (short):
   - Correlation drift, changing volatilities, transaction costs, liquidity, rebalancing cadence.
5) One-sentence takeaway the investor will remember.

Tone: professional, calm, concrete. Use the provided numbers; do not invent data.
"""
    return prompt.strip()
