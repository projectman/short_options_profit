import pytest
import pandas as pd
from options_analyzer.analyzer import (
    ShortOptionsAnalyzer,
    DiagonalSpreadAnalyzer,
    LongOptionPosition,
    StrategyRules,
    StrategyType,
    load_basis_long_positions_from_csv,
)


def test_load_basis_long_positions_from_csv():
    positions = load_basis_long_positions_from_csv("basis_long_positions.csv")
    symbols = [p.symbol for p in positions]
    assert "UPS" in symbols
    assert "XOM" in symbols
    assert "PLTR" in symbols

    pltr = next(p for p in positions if p.symbol == "PLTR")
    assert pltr.strike == 200.0
    assert pltr.cost_basis == 58.92
    assert pltr.expiration_date == "2027-06-17"


def test_strategy_rules_from_yaml():
    rules = StrategyRules.from_yaml("rules.yaml", positions_file="basis_long_positions.csv")
    assert rules.min_delta == 0.15
    assert rules.max_delta == 0.55
    assert rules.require_strike_less_than_spot is False
    assert rules.target_yield_diagonal == 0.80
    assert rules.target_yield_cash_protected == 0.50
    assert rules.target_yield_vertical == 0.50
    assert rules.vertical_target_delta_offset == 0.20
    assert rules.vertical_min_long_delta == 0.05
    
    symbols = rules.list_symbols()
    assert "UPS" in symbols
    assert "XOM" in symbols
    assert "PLTR" in symbols

    ups_pos = rules.get_basis_position("UPS")
    assert ups_pos is not None
    assert ups_pos.strike == 80.0
    assert ups_pos.cost_basis == 3.37
    assert ups_pos.expiration_date == "2027-06-17"


def test_diagonal_spread_analyzer_ups():
    rules = StrategyRules.from_yaml("rules.yaml")
    ups_pos = rules.get_basis_position("UPS")
    analyzer = ShortOptionsAnalyzer(basis_long=ups_pos, rules=rules)

    # Test candidate row: Strike 100 (OTM, Spot 104.50), Mid = 1.59
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
    assert result["strategy_type"] == "Diagonal Put Spread"
    assert result["strike"] == 100.0
    assert result["mid_price"] == 1.59
    
    # Full Profit = 100% of extrinsic = 1.59 * 100 = $159.00
    assert result["profit_usd"] == pytest.approx(159.00, rel=1e-2)
    
    # Target Profit = 80% of extrinsic = 1.59 * 0.80 * 100 = $127.20
    assert result["target_profit_usd"] == pytest.approx(127.20, rel=1e-2)
    
    # Spread risk: (100 - 80) + (3.37 - 1.59) = 20 + 1.78 = 21.78 -> $2178.00
    assert result["max_risk_usd"] == pytest.approx(2178.0, rel=1e-2)
    
    # Target Yield % = (127.20 / 2178.0) * 100 = 5.84%
    assert result["target_yield_pct"] == pytest.approx((127.20 / 2178.0) * 100, rel=1e-2)
    assert result["days_to_target"] > 0
    assert result["daily_relative_profit"] > 0
    assert result["expected_daily_relative_profit"] > 0


def test_cash_protected_put_analyzer_aapl():
    rules = StrategyRules.from_yaml("rules.yaml")
    analyzer = ShortOptionsAnalyzer(basis_long=None, rules=rules, strategy_type=StrategyType.CASH_PROTECTED_PUT)

    # AAPL 300.00P (Spot = 306.00, Mid = 7.10, IV = 0.2288, DTE = 33)
    row = pd.Series({
        "Strike": 300.0,
        "Bid": 7.00,
        "Ask": 7.20,
        "Mid": 7.10,
        "Delta": -0.4562,
        "IV": 0.2288,
        "dte": 33,
        "expiration_date": "2026-09-18",
        "symbol": "AAPL",
    })

    result = analyzer.analyze_candidate(row, spot_price=306.00)

    assert result["symbol"] == "AAPL"
    assert result["strategy_type"] == "Cash Protected Put"
    assert result["strike"] == 300.0
    assert result["mid_price"] == 7.10
    
    # Full Profit = 100% of extrinsic = 7.10 * 100 = $710.00
    assert result["profit_usd"] == pytest.approx(710.00, rel=1e-2)
    
    # Target Profit = 50% of extrinsic = 7.10 * 0.50 * 100 = $355.00
    assert result["target_profit_usd"] == pytest.approx(355.00, rel=1e-2)
    
    # Cash Protected Put Max Risk = (Strike - Mid) * 100 = (300.0 - 7.10) * 100 = $29290.00
    assert result["max_risk_usd"] == pytest.approx((300.0 - 7.10) * 100, rel=1e-2)
    
    # Target Yield % = (355.00 / 29290.00) * 100 = 1.21%
    assert result["target_yield_pct"] == pytest.approx((355.00 / 29290.00) * 100, rel=1e-2)
    assert result["days_to_target"] > 0
    assert result["daily_relative_profit"] > 0
    assert result["expected_daily_relative_profit"] > 0


def test_vertical_spread_analyzer_aapl():
    rules = StrategyRules.from_yaml("rules.yaml")
    analyzer = ShortOptionsAnalyzer(rules=rules, strategy_type=StrategyType.VERTICAL_SPREAD)

    # Mock options chain for AAPL
    df_chain = pd.DataFrame([
        {"Strike": 300.0, "Mid": 5.45, "Delta": -0.3622, "IV": 0.2281, "dte": 33, "expiration_date": "2026-09-18", "Type": "Put", "symbol": "AAPL"},
        {"Strike": 295.0, "Mid": 3.80, "Delta": -0.2772, "IV": 0.2301, "dte": 33, "expiration_date": "2026-09-18", "Type": "Put", "symbol": "AAPL"},
        {"Strike": 290.0, "Mid": 2.63, "Delta": -0.2062, "IV": 0.2350, "dte": 33, "expiration_date": "2026-09-18", "Type": "Put", "symbol": "AAPL"},
        {"Strike": 285.0, "Mid": 1.80, "Delta": -0.1496, "IV": 0.2411, "dte": 33, "expiration_date": "2026-09-18", "Type": "Put", "symbol": "AAPL"}, # target: 0.3622 - 0.20 = 0.1622 (closest: 0.1496!)
        {"Strike": 280.0, "Mid": 1.25, "Delta": -0.1080, "IV": 0.2498, "dte": 33, "expiration_date": "2026-09-18", "Type": "Put", "symbol": "AAPL"},
    ])

    short_row = df_chain.iloc[0]
    best_long = analyzer.find_vertical_long_put(short_row, df_chain)
    
    assert best_long is not None
    # 285.00P has delta -0.1496, which is closest to (0.3622 - 0.20 = 0.1622)
    assert best_long["Strike"] == 285.0

    result = analyzer.analyze_candidate(short_row, spot_price=306.00, full_df=df_chain)
    assert result is not None
    assert result["strategy_type"] == "Vertical Put Spread"
    assert result["strike"] == 300.0
    assert result["long_strike"] == 285.0
    
    # Net Credit = 5.45 - 1.80 = 3.65 -> Profit = $365.00
    assert result["profit_usd"] == pytest.approx(365.00, rel=1e-2)
    
    # Target Profit (50%) = 0.50 * 365.00 = $182.50
    assert result["target_profit_usd"] == pytest.approx(182.50, rel=1e-2)
    
    # Max Risk = (300 - 285 - 3.65) * 100 = 11.35 * 100 = $1135.00
    assert result["max_risk_usd"] == pytest.approx(1135.00, rel=1e-2)
    
    # Target Yield % = (182.50 / 1135.00) * 100 = 16.08%
    assert result["target_yield_pct"] == pytest.approx((182.50 / 1135.00) * 100, rel=1e-2)
