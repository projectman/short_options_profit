"""CLI Script to generate 3D Scatter & Line plots for vertical spreads and short options analysis."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from options_analyzer.plotter import plot_vertical_spread_3d


def main():
    parser = argparse.ArgumentParser(description="Generate 3D Scatter and Line plots for vertical spreads.")
    parser.add_argument("--symbol", default="AAPL", help="Ticker symbol to plot (default: AAPL)")
    parser.add_argument("--csv", default=None, help="Explicit path to CSV file (e.g. output/vertical_spread_analysis_AAPL.csv)")
    parser.add_argument("--output", default=None, help="Explicit output PNG file path")
    parser.add_argument("--elevation", type=float, default=25, help="3D plot elevation angle (default: 25)")
    parser.add_argument("--azimuth", type=float, default=-60, help="3D plot azimuth angle (default: -60)")
    args = parser.parse_args()

    sym = args.symbol.upper()
    csv_file = Path(args.csv) if args.csv else Path(f"output/vertical_spread_analysis_{sym}.csv")

    if not csv_file.exists():
        print(f"Error: CSV file '{csv_file}' not found. Please run analysis first with `python main.py --symbol {sym}`.")
        sys.exit(1)

    print(f"Generating 3D Scatter & Line plot for {sym} from {csv_file}...")
    saved_path = plot_vertical_spread_3d(
        csv_or_df=csv_file,
        output_png=args.output,
        symbol=sym,
        elevation=args.elevation,
        azimuth=args.azimuth,
    )
    print(f"✓ 3D Plot successfully generated and saved to: {saved_path}")


if __name__ == "__main__":
    main()
