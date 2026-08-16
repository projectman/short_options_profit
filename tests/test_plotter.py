import pytest
from pathlib import Path
import pandas as pd
from options_analyzer.plotter import plot_vertical_spread_3d


def test_plot_vertical_spread_3d(tmp_path):
    df_test = pd.DataFrame([
        {
            "symbol": "AAPL",
            "delta": -0.20,
            "expected_daily_relative_profit": 0.34,
            "target_yield_pct": 5.62,
            "strike": 290.0,
            "long_strike": 270.0,
            "expiration_date": "2026-09-18",
        },
        {
            "symbol": "AAPL",
            "delta": -0.36,
            "expected_daily_relative_profit": 0.53,
            "target_yield_pct": 16.08,
            "strike": 300.0,
            "long_strike": 285.0,
            "expiration_date": "2026-09-18",
        },
        {
            "symbol": "AAPL",
            "delta": -0.26,
            "expected_daily_relative_profit": 0.20,
            "target_yield_pct": 7.71,
            "strike": 290.0,
            "long_strike": 260.0,
            "expiration_date": "2026-10-16",
        },
    ])

    out_file = tmp_path / "test_3d_scatter.png"
    result_path = plot_vertical_spread_3d(df_test, output_png=out_file, symbol="AAPL")

    assert result_path.exists()
    assert result_path.stat().st_size > 1000  # valid PNG binary
