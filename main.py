"""Main entry point for Short Options Profit & Diagonal Spread Selection Analyzer."""

import sys
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.table import Table

# Add src to python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from options_analyzer.loader import DataLoader
from options_analyzer.analyzer import DiagonalSpreadAnalyzer, LongOptionPosition


def main():
    console = Console()
    console.print("\n[bold cyan]═══ Short Options Profit & Diagonal Spread Selection ═══[/bold cyan]\n")

    loader = DataLoader("source")
    files = loader.list_files()

    if not files:
        console.print("[yellow]No data files found in [bold]source/[/bold] folder.[/yellow]")
        console.print("Place your options CSV/Parquet files into [bold]source/[/bold] to begin analysis.")
        return

    # Load and combine all datasets from source/
    df_raw = loader.load_all()
    if df_raw.empty:
        console.print("[red]Could not load data from files in source/.[/red]")
        return

    # Determine underlying spot price
    spot_price = DataLoader.estimate_spot_price(df_raw)
    console.print(f"[bold green]Underlying Spot Price (UPS):[/bold green] [bold white]${spot_price:.2f}[/bold white]")

    # Configure Basis Long Position
    basis_long = LongOptionPosition(
        symbol="UPS",
        option_type="Put",
        strike=80.0,
        expiration_date="2027-06-17",
        cost_basis=3.37,
    )
    console.print(
        f"[bold yellow]Basis Long Put:[/bold yellow] {basis_long.symbol} ${basis_long.strike:.2f}P "
        f"Exp: {basis_long.expiration_date} | Cost: ${basis_long.cost_basis:.2f} / share ($337.00/contract)\n"
    )

    # Initialize Strategy Analyzer
    analyzer = DiagonalSpreadAnalyzer(
        basis_long=basis_long,
        target_yield=0.80,  # 80% extrinsic profit target (20% residual price)
        risk_free_rate=0.045,
    )

    results_df = analyzer.analyze_dataset(df_raw, spot_price=spot_price)

    if results_df.empty:
        console.print("[yellow]No short put candidates matched the filter criteria (Delta in 0.15 - 0.55, Strike < Spot).[/yellow]")
        return

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reorder columns as requested:
    # 1. delta (sorted by this value)
    # 2. short_put_index
    # 3. daily_relative_profit
    # 4. days_to_target
    # 5. profit_usd
    # followed by supporting columns for analysis
    display_cols = [
        "delta",
        "short_put_index",
        "daily_relative_profit",
        "days_to_target",
        "profit_usd",
        "strike",
        "expiration_date",
        "dte",
        "mid_price",
        "spread_risk_usd",
        "daily_profit_usd",
        "iv_pct",
    ]
    export_df = results_df[display_cols].copy()

    # Save to CSV and Markdown
    csv_path = output_dir / "diagonal_spread_analysis.csv"
    md_path = output_dir / "diagonal_spread_analysis.md"

    export_df.to_csv(csv_path, index=False)
    
    # Generate markdown table report
    md_content = f"""# Short Put Options Selection & Diagonal Spread Analysis

**Underlying**: UPS ($ {spot_price:.2f})  
**Basis Long Put**: UPS $80.00P Exp: 2027-06-17 (Cost: $3.37 / share)  
**Strategy**: Diagonal Put Spread with 80% Extrinsic Profit Target (20% residual value decay)  
**Valuation Rule**: Always use Medium price for Bid/Ask: $\\text{{Mid}} = \\frac{{\\text{{Bid}} + \\text{{Ask}}}}{{2}}$  

## Candidate Short Puts (Sorted by Delta)

| Delta | Short Put Identifier | Daily Rel Profit (%) | Days to Target | Profit ($) | Strike ($) | Expiration | DTE | Mid Price ($) | Spread Risk ($) | Daily Profit ($) | IV (%) |
|-------|----------------------|----------------------|----------------|------------|------------|------------|-----|---------------|-----------------|------------------|--------|
"""
    for _, row in export_df.iterrows():
        md_content += (
            f"| {row['delta']:+.4f} | `{row['short_put_index']}` | **{row['daily_relative_profit']:.3f}%** | "
            f"{row['days_to_target']:.1f} | ${row['profit_usd']:.2f} | ${row['strike']:.2f} | "
            f"{row['expiration_date']} | {row['dte']} | ${row['mid_price']:.2f} | "
            f"${row['spread_risk_usd']:.2f} | ${row['daily_profit_usd']:.2f} | {row['iv_pct']:.1f}% |\n"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Display Rich Table
    table = Table(
        title="Candidate Short Puts for Diagonal Spread (Sorted by Delta)",
        title_style="bold magenta",
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Delta", justify="right", style="cyan")
    table.add_column("Short Put Index", justify="left", style="bold white")
    table.add_column("Daily Rel Profit", justify="right", style="bold green")
    table.add_column("Days to Target", justify="right", style="yellow")
    table.add_column("Profit ($)", justify="right", style="green")
    table.add_column("Strike", justify="right")
    table.add_column("Exp Date", justify="center")
    table.add_column("DTE", justify="right")
    table.add_column("Mid ($)", justify="right")
    table.add_column("Spread Risk ($)", justify="right", style="red")

    for _, row in export_df.iterrows():
        table.add_row(
            f"{row['delta']:+.4f}",
            str(row["short_put_index"]),
            f"{row['daily_relative_profit']:.3f}%",
            f"{row['days_to_target']:.1f}",
            f"${row['profit_usd']:.2f}",
            f"${row['strike']:.2f}",
            str(row["expiration_date"]),
            str(row["dte"]),
            f"${row['mid_price']:.2f}",
            f"${row['spread_risk_usd']:.2f}",
        )

    console.print(table)
    console.print(f"\n[bold green]✓[/bold green] Analysis complete! Results written to:")
    console.print(f"  • [cyan]{csv_path}[/cyan]")
    console.print(f"  • [cyan]{md_path}[/cyan]\n")


if __name__ == "__main__":
    main()
