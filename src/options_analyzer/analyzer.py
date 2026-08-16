"""Unified Options Strategy Analyzer for Diagonal Put Spreads and Cash Protected Puts (Cash Secured Puts).

Single codebase providing high code reuse for pricing, decay, profit yield, and risk calculations.
Strategy-specific profit targets:
  - Diagonal Put Spreads: 80% target yield (decay to 20% residual extrinsic)
  - Cash Protected Short Puts: 50% target yield (decay to 50% residual extrinsic)
Rules and constants are loaded from rules.yaml.
Basis long positions for diagonal spreads are loaded from basis_long_positions.csv.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import yaml

from options_analyzer.black_scholes import find_days_to_target_put

logger = logging.getLogger("options_analyzer")


class StrategyType(str, Enum):
    """Supported short options strategy types."""
    DIAGONAL_SPREAD = "diagonal_spread"
    CASH_PROTECTED_PUT = "cash_protected_put"


@dataclass
class LongOptionPosition:
    """Data structure representing a basis long option position loaded from CSV."""
    symbol: str
    option_type: str
    strike: float
    expiration_date: str
    cost_basis: float
    contracts: int = 1


def load_basis_long_positions_from_csv(csv_path: Union[str, Path] = "basis_long_positions.csv") -> List[LongOptionPosition]:
    """Load basis long option positions directly from a CSV file."""
    path = Path(csv_path)
    if not path.exists():
        return []

    df = pd.read_csv(path)
    required_cols = {"symbol", "strike", "expiration_date", "cost_basis"}
    missing = required_cols - {c.lower() for c in df.columns}
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    # Standardize column headers to lowercase
    col_map = {c: c.lower().strip() for c in df.columns}
    df_clean = df.rename(columns=col_map)

    positions: List[LongOptionPosition] = []
    for _, row in df_clean.iterrows():
        pos = LongOptionPosition(
            symbol=str(row["symbol"]).upper().strip(),
            option_type=str(row.get("option_type", "Put")).strip(),
            strike=float(row["strike"]),
            expiration_date=str(row["expiration_date"]).strip(),
            cost_basis=float(row["cost_basis"]),
            contracts=int(row.get("contracts", 1)) if not pd.isna(row.get("contracts", 1)) else 1,
        )
        positions.append(pos)

    return positions


@dataclass
class StrategyRules:
    """Strategy rules loaded from rules.yaml and basis long positions loaded from CSV."""
    min_delta: float
    max_delta: float
    require_strike_less_than_spot: bool
    option_type: str
    require_positive_mid: bool
    risk_free_rate: float
    target_yield_diagonal: float = 0.80
    target_residual_ratio_diagonal: float = 0.20
    target_yield_cash_protected: float = 0.50
    target_residual_ratio_cash_protected: float = 0.50
    basis_long_positions: List[LongOptionPosition] = field(default_factory=list)

    @property
    def target_yield(self) -> float:
        """Default target yield."""
        return self.target_yield_diagonal

    @property
    def target_residual_ratio(self) -> float:
        """Default target residual ratio."""
        return self.target_residual_ratio_diagonal

    def get_target_yield(self, strategy_type: Union[str, StrategyType]) -> float:
        """Get target profit yield for the given strategy type (0.80 for diagonal, 0.50 for cash protected put)."""
        st = str(strategy_type).lower()
        if "cash" in st:
            return self.target_yield_cash_protected
        return self.target_yield_diagonal

    def get_target_residual_ratio(self, strategy_type: Union[str, StrategyType]) -> float:
        """Get target residual ratio for the given strategy type (0.20 for diagonal, 0.50 for cash protected put)."""
        st = str(strategy_type).lower()
        if "cash" in st:
            return self.target_residual_ratio_cash_protected
        return self.target_residual_ratio_diagonal

    @property
    def basis_long(self) -> Optional[LongOptionPosition]:
        """Returns the first basis long position if available."""
        return self.basis_long_positions[0] if self.basis_long_positions else None

    def get_basis_position(self, symbol: str) -> Optional[LongOptionPosition]:
        """Find the basis long position for a given ticker symbol."""
        sym_clean = symbol.upper().strip()
        for pos in self.basis_long_positions:
            if pos.symbol.upper().strip() == sym_clean:
                return pos
        return None

    def list_symbols(self) -> List[str]:
        """List all symbols configured in basis_long_positions."""
        return [pos.symbol.upper() for pos in self.basis_long_positions]

    @classmethod
    def from_yaml(
        cls,
        path: Union[str, Path] = "rules.yaml",
        positions_file: Optional[Union[str, Path]] = None,
    ) -> "StrategyRules":
        """Load strategy configuration from YAML and basis positions from CSV."""
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
        profit_cfg = data.get("profit_target", {})

        # Support both nested strategy targets and flat targets
        if "diagonal_spread" in profit_cfg and isinstance(profit_cfg["diagonal_spread"], dict):
            ty_diag = float(profit_cfg["diagonal_spread"].get("target_yield", 0.80))
            tr_diag = float(profit_cfg["diagonal_spread"].get("target_residual_ratio", 0.20))
        else:
            ty_diag = float(profit_cfg.get("target_yield", 0.80))
            tr_diag = float(profit_cfg.get("target_residual_ratio", 0.20))

        if "cash_protected_put" in profit_cfg and isinstance(profit_cfg["cash_protected_put"], dict):
            ty_cash = float(profit_cfg["cash_protected_put"].get("target_yield", 0.50))
            tr_cash = float(profit_cfg["cash_protected_put"].get("target_residual_ratio", 0.50))
        else:
            ty_cash = 0.50
            tr_cash = 0.50

        # Determine positions file path
        pos_csv = positions_file or data.get("positions_file", "basis_long_positions.csv")
        pos_csv_path = Path(pos_csv)
        if not pos_csv_path.is_absolute():
            candidate_1 = file_path.parent / pos_csv
            candidate_2 = Path.cwd() / pos_csv
            pos_csv_path = candidate_1 if candidate_1.exists() else candidate_2

        basis_positions: List[LongOptionPosition] = []
        if pos_csv_path.exists():
            basis_positions = load_basis_long_positions_from_csv(pos_csv_path)
        else:
            # Fallback to embedded yaml definitions if any
            positions_raw = data.get("basis_long_positions") or data.get("basis_long")
            if isinstance(positions_raw, dict):
                positions_raw = [positions_raw]
            elif not isinstance(positions_raw, list):
                positions_raw = []

            for b_dict in positions_raw:
                pos = LongOptionPosition(
                    symbol=str(b_dict["symbol"]).upper().strip(),
                    option_type=str(b_dict.get("option_type", "Put")),
                    strike=float(b_dict["strike"]),
                    expiration_date=str(b_dict["expiration_date"]),
                    cost_basis=float(b_dict["cost_basis"]),
                    contracts=int(b_dict.get("contracts", 1)),
                )
                basis_positions.append(pos)

        return cls(
            min_delta=float(delta_cfg["min_delta"]),
            max_delta=float(delta_cfg["max_delta"]),
            require_strike_less_than_spot=bool(strike_cfg["require_strike_less_than_spot"]),
            option_type=str(opt_cfg["option_type"]),
            require_positive_mid=bool(opt_cfg["require_positive_mid"]),
            target_yield_diagonal=ty_diag,
            target_residual_ratio_diagonal=tr_diag,
            target_yield_cash_protected=ty_cash,
            target_residual_ratio_cash_protected=tr_cash,
            risk_free_rate=float(val_cfg["risk_free_rate"]),
            basis_long_positions=basis_positions,
        )


class ShortOptionsAnalyzer:
    """Unified options profitability engine for both Diagonal Put Spreads and Cash Protected Puts."""

    def __init__(
        self,
        basis_long: Optional[LongOptionPosition] = None,
        rules: Optional[StrategyRules] = None,
        strategy_type: Optional[Union[str, StrategyType]] = None,
        config_path: Union[str, Path] = "rules.yaml",
        positions_path: Optional[Union[str, Path]] = None,
    ):
        self.rules = rules or StrategyRules.from_yaml(config_path, positions_file=positions_path)
        self.basis_long = basis_long
        
        # Auto-detect strategy type if not explicitly provided
        if isinstance(strategy_type, StrategyType):
            self.strategy_type = strategy_type
        elif strategy_type is not None:
            self.strategy_type = StrategyType(str(strategy_type).lower())
        else:
            self.strategy_type = (
                StrategyType.DIAGONAL_SPREAD if self.basis_long is not None
                else StrategyType.CASH_PROTECTED_PUT
            )

        # Strategy-specific profit target parameters
        self.target_yield = self.rules.get_target_yield(self.strategy_type)
        self.target_residual_ratio = self.rules.get_target_residual_ratio(self.strategy_type)
        self.risk_free_rate = self.rules.risk_free_rate

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

    def inspect_and_filter_candidates(
        self,
        df: pd.DataFrame,
        spot_price: float,
        min_delta: Optional[float] = None,
        max_delta: Optional[float] = None,
        require_strike_less_than_spot: Optional[bool] = None,
        symbol: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filters candidate puts and returns both selected candidates and full diagnostics log."""
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        min_d = min_delta if min_delta is not None else self.rules.min_delta
        max_d = max_delta if max_delta is not None else self.rules.max_delta
        req_otm = require_strike_less_than_spot if require_strike_less_than_spot is not None else self.rules.require_strike_less_than_spot
        
        default_sym = self.basis_long.symbol if self.basis_long else "UNKNOWN"
        target_symbol = (symbol or default_sym).upper().strip()

        data = df.copy()

        # Filter by symbol if symbol column is present
        if "symbol" in data.columns and target_symbol != "UNKNOWN":
            data = data[data["symbol"].str.upper() == target_symbol].copy()

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
                "symbol": target_symbol,
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
        """Perform comprehensive pricing decay, risk, Full Profit, Target Profit, and Target Yield calculations."""
        strike = float(row["Strike"])
        mid_price = float(row["Mid"])
        iv = float(row["IV"]) if ("IV" in row and not pd.isna(row["IV"]) and row["IV"] > 0) else 0.25
        dte_days = float(row["dte"]) if "dte" in row and not pd.isna(row["dte"]) else 30.0
        delta = float(row["Delta"])
        abs_delta = abs(delta)
        exp_date = str(row.get("expiration_date", "Unknown"))
        
        default_sym = self.basis_long.symbol if self.basis_long else "UNKNOWN"
        symbol = str(row.get("symbol", default_sym)).upper()

        # Intrinsic & Extrinsic values
        intrinsic_value = max(0.0, strike - spot_price)
        extrinsic_value = max(0.0, mid_price - intrinsic_value)

        # Full Profit (100% of extrinsic premium collected, in USD)
        full_profit_usd = extrinsic_value * 100.0

        # Target Profit (configured yield: 80% for diagonal, 50% for cash protected put)
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

        # Nominal Daily profit in USD
        daily_profit_usd = target_profit_usd / effective_days

        # Risk calculation based on strategy type:
        if self.strategy_type == StrategyType.DIAGONAL_SPREAD and self.basis_long is not None:
            # Diagonal Put Spread Risk:
            # Max risk per share = (Short Strike - Long Strike) + (Long Cost Basis - Short Mid Premium)
            strike_diff = max(0.0, strike - self.basis_long.strike)
            net_debit = self.basis_long.cost_basis - mid_price
            spread_risk_per_share = strike_diff + net_debit
            max_risk_usd = max(1.0, spread_risk_per_share * 100.0)
            strategy_name = "Diagonal Put Spread"
        else:
            # Cash Protected Put Risk:
            # Max risk per share = Strike - Mid Price (Strike minus cost of PUT)
            max_risk_per_share = max(0.01, strike - mid_price)
            max_risk_usd = max_risk_per_share * 100.0
            strategy_name = "Cash Protected Put"

        # Target Yield % = (Target Profit / Max Risk) * 100
        target_yield_pct = (target_profit_usd / max_risk_usd) * 100.0

        # Nominal Daily Relative Profit % = (daily_profit_usd / max_risk_usd) * 100
        nominal_daily_rel_profit_pct = (daily_profit_usd / max_risk_usd) * 100.0

        # Estimated Probability of Profit (P_win = 1 - |Delta|)
        p_win = max(0.0, min(1.0, 1.0 - abs_delta))

        # Expected Daily Relative Profit % = P_win * Nominal Daily Relative Profit %
        expected_daily_rel_profit_pct = p_win * nominal_daily_rel_profit_pct

        # Delta Efficiency = Nominal Daily Relative Profit % / |Delta|
        delta_efficiency = (nominal_daily_rel_profit_pct / abs_delta) if abs_delta > 0 else 0.0

        # Short put index/identifier format: SYMBOL YYMMDD P STRIKE
        short_put_index = f"{symbol} {exp_date} {strike:.2f}P"

        return {
            "symbol": symbol,
            "strategy_type": strategy_name,
            "delta": delta,
            "abs_delta": abs_delta,
            "short_put_index": short_put_index,
            "expected_daily_relative_profit": round(expected_daily_rel_profit_pct, 3),
            "daily_relative_profit": round(nominal_daily_rel_profit_pct, 3),
            "days_to_target": round(days_to_target, 2),
            "profit_usd": round(full_profit_usd, 2),            # Full 100% Extrinsic Profit in USD
            "max_risk_usd": round(max_risk_usd, 2),            # Max Risk in USD
            "target_profit_usd": round(target_profit_usd, 2),  # Target Profit (50% or 80% Extrinsic) in USD
            "target_yield_pct": round(target_yield_pct, 2),    # Target Yield % = Target Profit / Max Risk
            "p_win_pct": round(p_win * 100, 2),
            "strike": strike,
            "expiration_date": exp_date,
            "dte": int(dte_days),
            "mid_price": round(mid_price, 4),
            "iv_pct": round(iv * 100, 2),
            "daily_profit_usd": round(daily_profit_usd, 2),
            "delta_efficiency": round(delta_efficiency, 3),
            "extrinsic_value": round(extrinsic_value, 4),
            "target_price": round(target_price, 4),
            "target_yield_ratio": self.target_yield,
            "spread_risk_usd": round(max_risk_usd, 2),          # for backward compatibility
        }

    def analyze_dataset(
        self,
        df: pd.DataFrame,
        spot_price: float,
        min_delta: Optional[float] = None,
        max_delta: Optional[float] = None,
        require_strike_less_than_spot: Optional[bool] = None,
        symbol: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Analyze all valid candidate short puts in dataset, returning (results_df, diagnostics_df)."""
        candidates, diag_df = self.inspect_and_filter_candidates(
            df,
            spot_price=spot_price,
            min_delta=min_delta,
            max_delta=max_delta,
            require_strike_less_than_spot=require_strike_less_than_spot,
            symbol=symbol,
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


# Alias for backward compatibility
DiagonalSpreadAnalyzer = ShortOptionsAnalyzer
