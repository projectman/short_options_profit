"""Black-Scholes analytical pricing models, Greeks, and numerical solver for target price decay."""

import math
from typing import Optional
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def black_scholes_put_price(
    spot: float,
    strike: float,
    dte_years: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Calculate the theoretical Black-Scholes price for a European Put Option."""
    if dte_years <= 0:
        return max(0.0, strike - spot)

    if volatility <= 1e-6:
        # Near zero volatility limit
        discount = math.exp(-risk_free_rate * dte_years)
        forward = spot * math.exp((risk_free_rate - dividend_yield) * dte_years)
        return discount * max(0.0, strike - forward)

    d1 = (
        math.log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * dte_years
    ) / (volatility * math.sqrt(dte_years))
    d2 = d1 - volatility * math.sqrt(dte_years)

    put_price = strike * math.exp(-risk_free_rate * dte_years) * norm.cdf(-d2) - spot * math.exp(
        -dividend_yield * dte_years
    ) * norm.cdf(-d1)
    return max(0.0, float(put_price))


def black_scholes_put_delta(
    spot: float,
    strike: float,
    dte_years: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Calculate the Black-Scholes Put Delta."""
    if dte_years <= 0:
        return -1.0 if spot < strike else 0.0

    d1 = (
        math.log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * dte_years
    ) / (volatility * math.sqrt(dte_years))
    return float(-math.exp(-dividend_yield * dte_years) * norm.cdf(-d1))


def find_days_to_target_put(
    spot: float,
    strike: float,
    current_dte_days: float,
    target_price: float,
    volatility: float,
    risk_free_rate: float = 0.045,
    dividend_yield: float = 0.0,
) -> float:
    """Find the number of elapsed days until the put option decays to target_price.

    Assuming spot price, volatility, and interest rates remain constant.
    Returns elapsed days: (current_dte_days - remaining_dte_days).
    """
    if current_dte_days <= 0:
        return 0.0

    # If target price is 0 or less, it decays at expiration
    if target_price <= 0:
        return float(current_dte_days)

    current_price = black_scholes_put_price(
        spot=spot,
        strike=strike,
        dte_years=current_dte_days / 365.0,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )

    # If already below or equal to target price
    if current_price <= target_price:
        return 0.0

    # At expiration (dte=0), for an OTM put (spot > strike), payoff is 0.0 <= target_price
    expiry_price = max(0.0, strike - spot)
    if expiry_price > target_price:
        # If ITM at expiry and intrinsic > target_price, it never reaches target without spot movement
        # Return total DTE as fallback
        return float(current_dte_days)

    # Objective function: price(remaining_days) - target_price = 0
    def objective(remaining_days: float) -> float:
        p = black_scholes_put_price(
            spot=spot,
            strike=strike,
            dte_years=remaining_days / 365.0,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            dividend_yield=dividend_yield,
        )
        return p - target_price

    try:
        # Root finding for remaining_days in [0.0001, current_dte_days]
        remaining_days_target = brentq(objective, 1e-5, current_dte_days)
        elapsed_days = current_dte_days - remaining_days_target
        return max(0.0, float(elapsed_days))
    except Exception:
        # Linear approximation fallback if root finding fails
        # Assuming quadratic or linear theta decay
        fraction_decay = max(0.0, min(1.0, (current_price - target_price) / (current_price - expiry_price + 1e-9)))
        return float(current_dte_days * fraction_decay)
