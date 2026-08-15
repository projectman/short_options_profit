import pytest
import pandas as pd
from options_analyzer.analyzer import DiagonalSpreadAnalyzer, LongOptionPosition


def test_diagonal_spread_analyzer():
    basis_long = LongOptionPosition(
        symbol="UPS",
        option_type="Put",
        strike=80.0,
        expiration_date="2027-06-17",
        cost_basis=3.37,
    )
    analyzer = DiagonalSpreadAnalyzer(basis_long=basis_long)

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
