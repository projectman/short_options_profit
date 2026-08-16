"""Interactive 3D Options Visualizer.

Opens pre-calculated CSV values in an interactive Matplotlib 3D window (with real-time rotation and zoom)
and exports a high-resolution PNG image.

Usage Examples:
    # Open interactive 3D window for AAPL vertical spread:
    python plot_3d.py --symbol AAPL --show

    # Select from available pre-calculated CSVs in output/ interactively:
    python plot_3d.py --show

    # Plot specific CSV file:
    python plot_3d.py --csv output/vertical_spread_analysis_AAPL.csv --show
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rich.console import Console
from rich.prompt import Prompt
from options_analyzer.plotter import plot_vertical_spread_3d


def list_available_output_csvs(output_dir: Path = Path("output")) -> List[Path]:
    """Find all pre-calculated options analysis CSV files in output/."""
    if not output_dir.exists():
        return []
    return sorted(list(output_dir.glob("*_analysis_*.csv")))


def main():
    parser = argparse.ArgumentParser(description="Open and display pre-calculated options spreads in an interactive Matplotlib 3D window.")
    parser.add_argument("--symbol", default=None, help="Ticker symbol (e.g. AAPL, PLTR, UPS, XOM)")
    parser.add_argument("--csv", default=None, help="Path to pre-calculated CSV file (e.g. output/vertical_spread_analysis_AAPL.csv)")
    parser.add_argument("--output", default=None, help="Output PNG path (optional)")
    parser.add_argument("--show", action="store_true", default=True, help="Open interactive Matplotlib GUI window (default: True)")
    parser.add_argument("--no-show", dest="show", action="store_false", help="Do not open window, only save PNG image")
    parser.add_argument("--elevation", type=float, default=25, help="3D plot elevation angle (default: 25)")
    parser.add_argument("--azimuth", type=float, default=-60, help="3D plot azimuth angle (default: -60)")
    args = parser.parse_args()

    console = Console()
    console.print("\n[bold cyan]═══ Interactive 3D Options Spread Visualizer ═══[/bold cyan]\n")

    output_dir = Path("output")
    available_csvs = list_available_output_csvs(output_dir)

    selected_csv: Path
    if args.csv:
        selected_csv = Path(args.csv)
        if not selected_csv.exists():
            console.print(f"[red]Error: CSV file '{selected_csv}' does not exist.[/red]")
            sys.exit(1)
    elif args.symbol:
        sym = args.symbol.upper()
        # Look for vertical_spread, diagonal_spread, or cash_protected_put
        candidates = [
            output_dir / f"vertical_spread_analysis_{sym}.csv",
            output_dir / f"diagonal_spread_analysis_{sym}.csv",
            output_dir / f"cash_protected_put_analysis_{sym}.csv",
        ]
        found = [p for p in candidates if p.exists()]
        if not found:
            console.print(f"[red]No pre-calculated CSV found for symbol {sym} in {output_dir}/.[/red]")
            console.print(f"Run `python main.py --symbol {sym}` first to calculate values.")
            sys.exit(1)
        selected_csv = found[0]
    else:
        if not available_csvs:
            console.print(f"[yellow]No pre-calculated CSV files found in {output_dir}/.[/yellow]")
            console.print("Run `python main.py --symbol ALL` first to generate data.")
            sys.exit(1)

        console.print("[bold green]Available Pre-Calculated Reports in output/:[/bold green]")
        for i, p in enumerate(available_csvs, 1):
            console.print(f"  [{i}] [cyan]{p.name}[/cyan]")

        default_idx = "1"
        # Find AAPL vertical spread if available as default
        for i, p in enumerate(available_csvs, 1):
            if "vertical_spread_analysis_AAPL" in p.name:
                default_idx = str(i)
                break

        choice = Prompt.ask(
            "\nSelect CSV number to visualize in 3D",
            choices=[str(i) for i in range(1, len(available_csvs) + 1)],
            default=default_idx,
        )
        selected_csv = available_csvs[int(choice) - 1]

    console.print(f"Loading data from: [bold yellow]{selected_csv}[/bold yellow]...")

    if args.show:
        console.print("[green]Opening interactive Matplotlib 3D window (Click & drag to rotate, scroll to zoom)...[/green]")

    png_path = plot_vertical_spread_3d(
        csv_or_df=selected_csv,
        output_png=args.output,
        show_interactive=args.show,
        elevation=args.elevation,
        azimuth=args.azimuth,
    )
    console.print(f"✓ Saved PNG export to: [bold cyan]{png_path}[/bold cyan]\n")


if __name__ == "__main__":
    main()
