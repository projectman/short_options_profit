"""3D Visualization module for Options Spread analysis.

Plots 3D Scatter & Line plots showing:
  - X-axis: Delta (or |Delta|)
  - Y-axis: Expected Daily Relative Profit (%)
  - Z-axis (Vertical scale): Target Yield (%)
Traces continuous trajectories (ax.plot) per expiration cycle and adds scatter markers (ax.scatter).
Supports both interactive Matplotlib window display (plt.show()) and high-resolution PNG export.
"""

import os
from pathlib import Path
from typing import Optional, Union, List
import pandas as pd
import numpy as np

_mpl_cache = Path(__file__).resolve().parent.parent.parent / ".matplotlib_cache"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_vertical_spread_3d(
    csv_or_df: Union[str, Path, pd.DataFrame],
    output_png: Optional[Union[str, Path]] = None,
    symbol: str = "AAPL",
    show_labels: bool = True,
    show_interactive: bool = False,
    elevation: float = 25,
    azimuth: float = -60,
    dpi: int = 300,
) -> Path:
    """Generate a 3D Scatter & Line plot for Options Spreads.
    
    Parameters:
        csv_or_df: Path to CSV file or existing DataFrame.
        output_png: Path to save the PNG image. Defaults to output/vertical_spread_3d_{symbol}.png.
        symbol: Ticker symbol (used if inferred from data).
        show_labels: Whether to annotate individual strike points.
        show_interactive: If True, opens the interactive Matplotlib window allowing 3D rotation/zoom.
        elevation: 3D viewing elevation angle.
        azimuth: 3D viewing azimuth angle.
        dpi: Output resolution for saved PNG.
    """
    if isinstance(csv_or_df, (str, Path)):
        csv_path = Path(csv_or_df)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path.resolve()}")
        df = pd.read_csv(csv_path)
    else:
        df = csv_or_df.copy()

    if df.empty:
        raise ValueError("DataFrame is empty, cannot generate 3D plot.")

    # Infer symbol and strategy
    if "symbol" in df.columns and len(df["symbol"].dropna()) > 0:
        sym = str(df["symbol"].dropna().iloc[0]).upper()
    else:
        sym = symbol.upper()

    strategy = "Spread / Put"
    if "strategy_type" in df.columns and len(df["strategy_type"].dropna()) > 0:
        strategy = str(df["strategy_type"].dropna().iloc[0])

    # Determine output path
    if output_png is None:
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = "vertical_spread_3d_scatter" if "vertical" in strategy.lower() else "options_3d_scatter"
        output_png = output_dir / f"{prefix}_{sym}.png"
    else:
        output_png = Path(output_png)
        output_png.parent.mkdir(parents=True, exist_ok=True)

    # Set styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(13, 9), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    # Column mapping with fallback
    x_col = "delta" if "delta" in df.columns else df.columns[0]
    y_col = "expected_daily_relative_profit" if "expected_daily_relative_profit" in df.columns else df.columns[1]
    z_col = "target_yield_pct" if "target_yield_pct" in df.columns else df.columns[2]

    # Separate curves by expiration date to draw continuous 3D trajectories (ax.plot)
    exp_dates = df["expiration_date"].unique() if "expiration_date" in df.columns else ["All"]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(1, len(exp_dates))))

    # Scatter points across all data for colorbar
    scatter = ax.scatter(
        df[x_col],
        df[y_col],
        df[z_col],
        c=df[z_col],
        cmap="viridis",
        s=85,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
        depthshade=True,
        label="Spread Candidates",
    )

    # Trace continuous 3D lines (ax.plot) per expiration date
    for i, exp in enumerate(exp_dates):
        if "expiration_date" in df.columns:
            sub_df = df[df["expiration_date"] == exp].copy()
        else:
            sub_df = df.copy()

        # Sort by delta for a continuous parametric trajectory curve
        sub_df = sub_df.sort_values(by=x_col)
        
        ax.plot(
            sub_df[x_col],
            sub_df[y_col],
            sub_df[z_col],
            label=f"Exp: {exp} Trajectory",
            color=colors[i],
            linewidth=2.2,
            linestyle="-",
            marker="o",
            markersize=5,
            alpha=0.85,
        )

        # Draw vertical drop stems down to z_min for 3D depth perception
        z_min = df[z_col].min() * 0.8
        for _, row in sub_df.iterrows():
            ax.plot(
                [row[x_col], row[x_col]],
                [row[y_col], row[y_col]],
                [z_min, row[z_col]],
                color="gray",
                linestyle=":",
                linewidth=0.7,
                alpha=0.45,
            )

    # Annotate key points with Short/Long strike names
    if show_labels:
        for _, row in df.iterrows():
            if "long_strike" in row and pd.notna(row["long_strike"]) and float(row["long_strike"]) > 0:
                strike_str = f"{row.get('strike', 0):.0f}P/{row.get('long_strike', 0):.0f}P"
            else:
                strike_str = f"{row.get('strike', 0):.0f}P"

            ax.text(
                row[x_col],
                row[y_col],
                row[z_col] + 0.6,
                strike_str,
                fontsize=8,
                color="#1a202c",
                weight="bold",
            )

    # Axis Labels & Title
    ax.set_xlabel(r"Short Put Delta ($\Delta$)", fontsize=11, labelpad=10, fontweight="bold")
    ax.set_ylabel("Expected Daily Rel Profit (%)", fontsize=11, labelpad=10, fontweight="bold")
    ax.set_zlabel("Target Yield (% on Max Risk)", fontsize=11, labelpad=10, fontweight="bold")
    
    ax.set_title(
        f"3D Trajectory & Scatter Analysis: {sym} {strategy}\n"
        r"$X=\Delta$, $Y=\mathbb{E}[\text{Daily Rel Profit}]$, $Z=\text{Target Yield (\%)} = \frac{\text{Target Profit}}{\text{Max Risk}}$",
        fontsize=13,
        pad=18,
        fontweight="bold",
    )

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.65, aspect=15)
    cbar.set_label("Target Yield (%)", fontsize=10, fontweight="bold")

    # Set viewing angle
    ax.view_init(elev=elevation, azim=azimuth)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")

    if show_interactive:
        # Open interactive Matplotlib GUI window allowing rotation/zoom
        plt.show()

    plt.close(fig)
    return output_png
