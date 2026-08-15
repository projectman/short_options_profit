import pytest
import pandas as pd
from options_analyzer.analyzer import DiagonalSpreadAnalyzer, LongOptionPosition, StrategyRules


def test_strategy_rules_from_yaml():
    rules = StrategyRules.from_yaml("rules.yaml")
    assert rules.min_delta == 0.10
    assert rules.max_delta == 0.55
    assert rules.require_strike_less_than_spot is False
    assert rules.basis_long.symbol == "UPS"
    assert rules.basis_long.strike == 80.0
    assert rules.basis_long.cost_basis == 3.37
    assert rules.basis_long.expiration_date == "2027-06-17"


def test_diagonal_spread_analyzer():
    rules = StrategyRules.from_yaml("rules.yaml")
    analyzer = DiagonalSpreadAnalyzer(rules=rules)

    # Test candidate row
    row = pd.Series({
        "Strike": 100.0,
        "Bid": 1.51,
        "Ask": 1.67,
        "Mid": 1.59,
        "Delta": -0.2873,
        "IV": 0.2655,
        "dte": 34,
        "expiration_date": "2026-09-18",
        "symbol": "UPS",
    })

    result = analyzer.analyze_candidate(row, spot_price=104.50)

    assert result["strike"] == 100.0
    assert result["mid_price"] == 1.59
    assert result["profit_usd"] == pytest.approx(1.59 * 0.80 * 100, rel=1e-2)
    # Spread risk: (100 - 80) + (3.37 - 1.59) = 20 + 1.78 = 21.78 -> $2178.00
    assert result["spread_risk_usd"] == pytest.approx(2178.0, rel=1e-2)
    assert result["days_to_target"] > 0
    assert result["daily_relative_profit"] > 0
