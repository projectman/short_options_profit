import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from options_analyzer.loader import DataLoader
from options_analyzer.analyzer import ShortOptionsAnalyzer


def test_data_loader_list_files():
    loader = DataLoader("source")
    files = loader.list_files()
    assert isinstance(files, list)


def test_short_put_payoff():
    # Spot = 100, Strike = 95, Premium = 3.0 -> Max profit = 3.0
    profit = ShortOptionsAnalyzer.calculate_short_put_payoff(spot_price=100.0, strike_price=95.0, premium=3.0)
    assert profit == 3.0

    # Spot = 90, Strike = 95, Premium = 3.0 -> Payoff = 3.0 - (95 - 90) = -2.0
    profit = ShortOptionsAnalyzer.calculate_short_put_payoff(spot_price=90.0, strike_price=95.0, premium=3.0)
    assert profit == -2.0


def test_short_call_payoff():
    # Spot = 100, Strike = 105, Premium = 2.5 -> Max profit = 2.5
    profit = ShortOptionsAnalyzer.calculate_short_call_payoff(spot_price=100.0, strike_price=105.0, premium=2.5)
    assert profit == 2.5

    # Spot = 110, Strike = 105, Premium = 2.5 -> Payoff = 2.5 - (110 - 105) = -2.5
    profit = ShortOptionsAnalyzer.calculate_short_call_payoff(spot_price=110.0, strike_price=105.0, premium=2.5)
    assert profit == -2.5
