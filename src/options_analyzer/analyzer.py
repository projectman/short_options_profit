"""Diagonal spread analyzer and short options profitability engine."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from options_analyzer.black_scholes import find_days_to_target_put, black_scholes_put_price


class ShortOptionsAnalyzer:
    """Base helper methods for standalone short options calculations."""

    @staticmethod
    def calculate_short_put_payoff(spot_price: float, strike_price: float, premium: float) -> float:
        """Calculate payoff at expiration for a short put option per share."""
        intrinsic_loss = max(0.0, strike_price - spot_price)
        return premium - intrinsic_loss

    @staticmethod
    def calculate_short_call_payoff(spot_price: float, strike_price: float, premium: float) -> float:
        """Calculate payoff at expiration for a short call option per share."""
        intrinsic_loss = max(0.0, spot_price - strike_price)
        return premium - intrinsic_loss


@dataclass
class LongOptionPosition:
    """Represents the basis long option position."""
    symbol: str = "UPS"
    option_type: str = "Put"
    strike: float = 80.0
    expiration_date: str = "2027-06-17"
    cost_basis: float = 3.37  # Purchase price per share ($3.37)
    contracts: int = 1


class DiagonalSpreadAnalyzer:
    """Analyzes candidate short puts paired with a basis long put for diagonal spread profitability."""

    def __init__(
        self,
        basis_long: Optional[LongOptionPosition] = None,
        target_yield: float = 0.80,
        risk_free_rate: float = 0.045,
    ):
        self.basis_long = basis_long or LongOptionPosition()
        self.target_yield = target_yield
        self.target_residual_ratio = 1.0 - target_yield  # 0.20 (20% of initial mid price)
        self.risk_free_rate = risk_free_rate

    def filter_short_put_candidates(
        self, df: pd.DataFrame, spot_price: float, min_delta: float = 0.15, max_delta: float = 0.55
    ) -> pd.DataFrame:
        """Filter for put options with strike below spot and absolute delta between min_delta and max_delta."""
        if df.empty:
            return pd.DataFrame()

        data = df.copy()

        # Filter by option type: Put
        if "Type" in data.columns:
            data = data[data["Type"].str.lower() == "put"]

        # Ensure numeric delta
        if "Delta" in data.columns:
            data["abs_delta"] = data["Delta"].abs()
        else:
            return pd.DataFrame()

        # Condition 1: Strike below stock price (OTM puts)
        data = data[data["Strike"] < spot_price]

        # Condition 2: Delta in [min_delta, max_delta]
        data = data[(data["abs_delta"] >= min_delta) & (data["abs_delta"] <= max_delta)]

        # Must have valid positive Mid price
        data = data[data["Mid"] > 0]

        return data.copy()

    def analyze_candidate(self, row: pd.Series, spot_price: float) -> Dict[str, Any]:
        """Perform comprehensive pricing decay, diagonal spread risk, and daily return calculations for a short put."""
        strike = float(row["Strike"])
        mid_price = float(row["Mid"])
        iv = float(row["IV"]) if ("IV" in row and not pd.isna(row["IV"]) and row["IV"] > 0) else 0.25
        dte_days = float(row["dte"]) if "dte" in row and not pd.isna(row["dte"]) else 30.0
        delta = float(row["Delta"])
        abs_delta = abs(delta)
        exp_date = str(row.get("expiration_date", "Unknown"))
        symbol = str(row.get("symbol", self.basis_long.symbol))

        # Intrinsic & Extrinsic values (OTM put has 0 intrinsic)
        intrinsic_value = max(0.0, strike - spot_price)
        extrinsic_value = max(0.0, mid_price - intrinsic_value)

        # Target profit: 80% of extrinsic value
        target_profit_per_share = self.target_yield * extrinsic_value
        target_profit_usd = target_profit_per_share * 100.0  # 1 contract = 100 shares

        # Target option price (when option decays to 20% of current mid price)
        target_price = self.target_residual_ratio * mid_price

        # Calculate days required to reach target price via theta decay (holding spot & IV constant)
        days_to_target = find_days_to_target_put(
            spot=spot_price,
            strike=strike,
            current_dte_days=dte_days,
            target_price=target_price,
            volatility=iv,
            risk_free_rate=self.risk_free_rate,
        )

        # Avoid zero division
        effective_days = max(1.0, days_to_target)

        # Daily profit in USD
        daily_profit_usd = target_profit_usd / effective_days

        # Risk calculation for diagonal spread on expiration date:
        # Max risk per share = (Short Strike - Long Strike) + (Long Cost Basis - Short Mid Premium)
        # For UPS 80P basis: (Strike - 80) + (3.37 - Mid)
        strike_diff = max(0.0, strike - self.basis_long.strike)
        net_debit = self.basis_long.cost_basis - mid_price
        spread_risk_per_share = strike_diff + net_debit
        spread_risk_usd = max(1.0, spread_risk_per_share * 100.0)

        # Daily Relative Profit % = (daily_profit_usd / spread_risk_usd) * 100
        daily_relative_profit_pct = (daily_profit_usd / spread_risk_usd) * 100.0

        # Short put index/identifier format: SYMBOL YYMMDD P STRIKE
        short_put_index = f"{symbol} {exp_date} {strike:.2f}P"

        return {
            "delta": delta,
            "abs_delta": abs_delta,
            "short_put_index": short_put_index,
            "daily_relative_profit": daily_relative_profit_pct,
            "days_to_target": round(days_to_target, 2),
            "profit_usd": round(target_profit_usd, 2),
            "strike": strike,
            "expiration_date": exp_date,
            "dte": int(dte_days),
            "mid_price": round(mid_price, 4),
            "iv_pct": round(iv * 100, 2),
            "spread_risk_usd": round(spread_risk_usd, 2),
            "daily_profit_usd": round(daily_profit_usd, 2),
            "extrinsic_value": round(extrinsic_value, 4),
            "target_price": round(target_price, 4),
        }

    def analyze_dataset(self, df: pd.DataFrame, spot_price: float) -> pd.DataFrame:
        """Analyze all valid candidate short puts in the dataset and return sorted results."""
        candidates = self.filter_short_put_candidates(df, spot_price=spot_price)
        if candidates.empty:
            return pd.DataFrame()

        results = []
        for _, row in candidates.iterrows():
            metrics = self.analyze_candidate(row, spot_price=spot_price)
            results.append(metrics)

        result_df = pd.DataFrame(results)

        # Sort by Delta (ascending by absolute delta or signed delta)
        result_df = result_df.sort_values(by="abs_delta", ascending=True).reset_index(drop=True)

        return result_df
