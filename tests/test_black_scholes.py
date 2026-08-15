import pytest
from options_analyzer.black_scholes import (
    black_scholes_put_price,
    black_scholes_put_delta,
    find_days_to_target_put,
)


def test_black_scholes_put_price():
    # Spot 100, Strike 100, 1 year, r=0.05, sigma=0.20
    price = black_scholes_put_price(spot=100.0, strike=100.0, dte_years=1.0, risk_free_rate=0.05, volatility=0.20)
    assert 5.0 < price < 6.0


def test_black_scholes_put_delta():
    # ATM Put delta should be around -0.4 to -0.5
    delta = black_scholes_put_delta(spot=100.0, strike=100.0, dte_years=0.5, risk_free_rate=0.05, volatility=0.20)
    assert -0.6 < delta < -0.3


def test_find_days_to_target_put():
    # Spot 104.5, Strike 100, DTE 34 days, IV 0.2655
    spot = 104.5
    strike = 100.0
    dte_days = 34.0
    iv = 0.2655
    mid_price = black_scholes_put_price(
        spot=spot,
        strike=strike,
        dte_years=dte_days / 365.0,
        risk_free_rate=0.045,
        volatility=iv,
    )
    target_price = 0.20 * mid_price  # 20% of mid price (80% profit)
    
    elapsed_days = find_days_to_target_put(
        spot=spot,
        strike=strike,
        current_dte_days=dte_days,
        target_price=target_price,
        volatility=iv,
        risk_free_rate=0.045,
    )

    assert 0 < elapsed_days <= dte_days
