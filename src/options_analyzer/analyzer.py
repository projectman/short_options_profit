"""Diagonal spread analyzer and short options profitability engine.

Rules and basis option positions are loaded directly from rules.yaml (Single Source of Truth).
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import yaml

from options_analyzer.black_scholes import find_days_to_target_put

logger = logging.getLogger("options_analyzer")


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
    """Data structure representing the basis long option position loaded from configuration."""
    symbol: str
    option_type: str
    strike: float
    expiration_date: str
    cost_basis: float
    contracts: int = 1


@dataclass
class StrategyRules:
    """Strategy rules loaded directly from rules.yaml."""
    min_delta: float
    max_delta: float
    require_strike_less_than_spot: bool
    option_type: str
    require_positive_mid: bool
    target_yield: float
    target_residual_ratio: float
    risk_free_rate: float
    basis_long: LongOptionPosition

    @classmethod
    def from_yaml(cls, path: Union[str, Path] = "rules.yaml") -> "StrategyRules":
        """Load strategy configuration strictly from a YAML file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path.resolve()}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML configuration format in {file_path}")

        delta_cfg = data["delta_filter"]
        strike_cfg = data["strike_filter"]
        opt_cfg = data["option_selection"]
        val_cfg = data["valuation"]
        profit_cfg = data["profit_target"]
        basis_cfg = data["basis_long"]

        basis_long = LongOptionPosition(
            symbol=str(basis_cfg["symbol"]),
            option_type=str(basis_cfg["option_type"]),
            strike=float(basis_cfg["strike"]),
            expiration_date=str(basis_cfg["expiration_date"]),
            cost_basis=float(basis_cfg["cost_basis"]),
            contracts=int(basis_cfg.get("contracts", 1)),
        )

        return cls(
            min_delta=float(delta_cfg["min_delta"]),
            max_delta=float(delta_cfg["max_delta"]),
            require_strike_less_than_spot=bool(strike_cfg["require_strike_less_than_spot"]),
            option_type=str(opt_cfg["option_type"]),
            require_positive_mid=bool(opt_cfg["require_positive_mid"]),
            target_yield=float(profit_cfg["target_yield"]),
            target_residual_ratio=float(profit_cfg["target_residual_ratio"]),
            risk_free_rate=float(val_cfg["risk_free_rate"]),
            basis_long=basis_long,
        )


class DiagonalSpreadAnalyzer:
    """Analyzes candidate short puts paired with a basis long put for diagonal spread profitability."""

    def __init__(
        self,
        rules: Optional[StrategyRules] = None,
        config_path: Union[str, Path] = "rules.yaml",
    ):
        # Load directly from YAML if no rules object is explicitly provided
        self.rules = rules or StrategyRules.from_yaml(config_path)
        self.basis_long = self.rules.basis_long
        self.target_yield = self.rules.target_yield
        self.target_residual_ratio = self.rules.target_residual_ratio
        self.risk_free_rate = self.rules.risk_free_rate

    def inspect_and_filter_candidates(
        self,
        df: pd.DataFrame,
        spot_price: float,
        min_delta: Optional[float] = None,
        max_delta: Optional[float] = None,
        require_strike_less_than_spot: Optional[bool] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filters candidate puts and returns both selected candidates and full diagnostics log."""
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        min_d = min_delta if min_delta is not None else self.rules.min_delta
        max_d = max_delta if max_delta is not None else self.rules.max_delta
        req_otm = require_strike_less_than_spot if require_strike_less_than_spot is not None else self.rules.require_strike_less_than_spot

        data = df.copy()

        # Filter to target Option Type from YAML (e.g. Put)
        if "Type" in data.columns:
            puts = data[data["Type"].str.lower() == self.rules.option_type.lower()].copy()
        else:
            puts = data.copy()

        if puts.empty:
            return pd.DataFrame(), pd.DataFrame()

        if "Delta" in puts.columns:
            puts["abs_delta"] = puts["Delta"].abs()
        else:
            puts["abs_delta"] = np.nan

        diagnostic_records = []
        selected_records = []

        for idx, row in puts.iterrows():
            strike = float(row.get("Strike", 0.0))
            delta = float(row.get("Delta", 0.0)) if not pd.isna(row.get("Delta")) else 0.0
            abs_delta = abs(delta)
            mid = float(row.get("Mid", 0.0)) if not pd.isna(row.get("Mid")) else 0.0
            exp = str(row.get("expiration_date", "Unknown"))
            dte = int(row.get("dte", 0))

            reasons_excluded = []
            is_otm = strike < spot_price
            delta_in_range = min_d <= abs_delta <= max_d
            valid_mid = mid > 0 if self.rules.require_positive_mid else True

            # Evaluate filter conditions
            if req_otm and not is_otm:
                reasons_excluded.append(f"ITM/ATM (Strike {strike:.2f} >= Spot {spot_price:.2f})")
            if abs_delta < min_d:
                reasons_excluded.append(f"Delta |{delta:+.4f}| < {min_d:.2f}")
            elif abs_delta > max_d:
                reasons_excluded.append(f"Delta |{delta:+.4f}| > {max_d:.2f}")
            if not valid_mid:
                reasons_excluded.append("Mid Price <= 0")

            status = "SELECTED" if (len(reasons_excluded) == 0) else "EXCLUDED"
            reason_str = f"Meets criteria (Delta {min_d:.2f}-{max_d:.2f})" if status == "SELECTED" else "; ".join(reasons_excluded)

            rec = {
                "expiration_date": exp,
                "dte": dte,
                "strike": strike,
                "delta": delta,
                "abs_delta": abs_delta,
                "mid": mid,
                "status": status,
                "reason": reason_str,
            }
            diagnostic_records.append(rec)

            if status == "SELECTED":
                selected_records.append(row)

        diag_df = pd.DataFrame(diagnostic_records)
        selected_df = pd.DataFrame(selected_records) if selected_records else pd.DataFrame()

        return selected_df, diag_df

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

        # Intrinsic & Extrinsic values
        intrinsic_value = max(0.0, strike - spot_price)
        extrinsic_value = max(0.0, mid_price - intrinsic_value)

        # Target profit: configured yield (e.g. 80%) of extrinsic value
        target_profit_per_share = self.target_yield * extrinsic_value
        target_profit_usd = target_profit_per_share * 100.0  # 1 contract = 100 shares

        # Target option price (when option retains only configured residual extrinsic value + intrinsic value)
        target_price = intrinsic_value + self.target_residual_ratio * extrinsic_value

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
        # Using long put basis from YAML rules
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

    def analyze_dataset(
        self,
        df: pd.DataFrame,
        spot_price: float,
        min_delta: Optional[float] = None,
        max_delta: Optional[float] = None,
        require_strike_less_than_spot: Optional[bool] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Analyze all valid candidate short puts in dataset, returning (results_df, diagnostics_df)."""
        candidates, diag_df = self.inspect_and_filter_candidates(
            df,
            spot_price=spot_price,
            min_delta=min_delta,
            max_delta=max_delta,
            require_strike_less_than_spot=require_strike_less_than_spot,
        )
        if candidates.empty:
            return pd.DataFrame(), diag_df

        results = []
        for _, row in candidates.iterrows():
            metrics = self.analyze_candidate(row, spot_price=spot_price)
            results.append(metrics)

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values(by="abs_delta", ascending=True).reset_index(drop=True)

        return result_df, diag_df
