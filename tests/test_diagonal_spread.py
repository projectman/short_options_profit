import pytest
import pandas as pd
from options_analyzer.analyzer import DiagonalSpreadAnalyzer, LongOptionPosition, StrategyRules


def test_strategy_rules_from_yaml():
    rules = StrategyRules.from_yaml("rules.yaml")
    assert rules.min_delta == 0.15
    assert rules.max_delta == 0.55
    assert rules.require_strike_less_than_spot is False
    
    symbols = rules.list_symbols()
    assert "UPS" in symbols
    assert "XOM" in symbols

    ups_pos = rules.get_basis_position("UPS")
    assert ups_pos is not None
    assert ups_pos.strike == 80.0
    assert ups_pos.cost_basis == 3.37
    assert ups_pos.expiration_date == "2027-06-17"

    xom_pos = rules.get_basis_position("XOM")
    assert xom_pos is not None
    assert xom_pos.strike == 100.0
    assert xom_pos.expiration_date == "2027-06-17"


def test_diagonal_spread_analyzer_ups():
    rules = StrategyRules.from_yaml("rules.yaml")
    ups_pos = rules.get_basis_position("UPS")
    analyzer = DiagonalSpreadAnalyzer(basis_long=ups_pos, rules=rules)

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

    assert result["symbol"] == "UPS"
    assert result["strike"] == 100.0
    assert result["mid_price"] == 1.59
    assert result["profit_usd"] == pytest.approx(1.59 * 0.80 * 100, rel=1e-2)
    # Spread risk: (100 - 80) + (3.37 - 1.59) = 20 + 1.78 = 21.78 -> $2178.00
    assert result["spread_risk_usd"] == pytest.approx(2178.0, rel=1e-2)
    assert result["days_to_target"] > 0
    
    # Expected Daily Relative Profit = (1 - 0.2873) * nominal_daily_profit%
    p_win = 1.0 - 0.2873
    assert result["p_win_pct"] == pytest.approx(p_win * 100, rel=1e-2)
    assert result["expected_daily_relative_profit"] > 0
    assert result["expected_daily_relative_profit"] < result["daily_profit_usd"] / result["spread_risk_usd"] * 100


def test_diagonal_spread_analyzer_xom():
    rules = StrategyRules.from_yaml("rules.yaml")
    xom_pos = rules.get_basis_position("XOM")
    analyzer = DiagonalSpreadAnalyzer(basis_long=xom_pos, rules=rules)

    row = pd.Series({
        "Strike": 150.0,
        "Bid": 1.65,
        "Ask": 1.82,
        "Mid": 1.735,
        "Delta": -0.2106,
        "IV": 0.2853,
        "dte": 34,
        "expiration_date": "2026-09-18",
        "symbol": "XOM",
    })

    result = analyzer.analyze_candidate(row, spot_price=158.0)

    assert result["symbol"] == "XOM"
    assert result["strike"] == 150.0
    assert result["profit_usd"] == pytest.approx(1.735 * 0.80 * 100, rel=1e-2)
    assert result["expected_daily_relative_profit"] > 0
