"""Main entry point for Short Options Profit & Diagonal Spread Selection Analyzer."""

import argparse
import sys
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add src to python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from options_analyzer.loader import DataLoader
from options_analyzer.analyzer import DiagonalSpreadAnalyzer, LongOptionPosition


def main():
    parser = argparse.ArgumentParser(description="Analyze diagonal spread short put candidates.")
    parser.add_argument("--source", default="source", help="Directory containing downloaded options files")
    parser.add_argument("--min-delta", type=float, default=0.15, help="Minimum absolute delta (default: 0.15)")
    parser.add_argument("--max-delta", type=float, default=0.55, help="Maximum absolute delta (default: 0.55)")
    parser.add_argument("--target-yield", type=float, default=0.80, help="Target extrinsic yield (default: 0.80)")
    parser.add_argument("--show-all-puts", action="store_true", help="Show full diagnostics of all scanned put options")
    args = parser.parse_args()

    console = Console()
    console.print("\n[bold cyan]═══ Short Options Profit & Diagonal Spread Selection ═══[/bold cyan]\n")

    loader = DataLoader(args.source)
    files = loader.list_files()

    if not files:
        console.print(f"[yellow]No data files found in [bold]{args.source}/[/bold] folder.[/yellow]")
        console.print(f"Place your options CSV/Parquet files into [bold]{args.source}/[/bold] to begin analysis.")
        return

    console.print(f"[bold blue]Step 1: Ingesting Data from {args.source}/[/bold blue]")
    for f in files:
        console.print(f"  • Found file: [cyan]{f.name}[/cyan] ({f.stat().st_size / 1024:.1f} KB)")

    df_raw = loader.load_all()
    if df_raw.empty:
        console.print("[red]Could not load data from files in source/.[/red]")
        return

    total_rows = len(df_raw)
    total_calls = len(df_raw[df_raw["Type"].str.lower() == "call"]) if "Type" in df_raw.columns else 0
    total_puts = len(df_raw[df_raw["Type"].str.lower() == "put"]) if "Type" in df_raw.columns else 0

    console.print(f"  → Loaded [bold]{total_rows}[/bold] total option contracts ([bold green]{total_calls} Calls[/bold green], [bold magenta]{total_puts} Puts[/bold magenta]).")

    # Determine underlying spot price
    spot_price = DataLoader.estimate_spot_price(df_raw)
    console.print(f"\n[bold blue]Step 2: Underlying Asset & Long Basis Setup[/bold blue]")
    console.print(f"  • Estimated Spot Price (UPS): [bold green]${spot_price:.2f}[/bold green]")

    # Configure Basis Long Position
    basis_long = LongOptionPosition(
        symbol="UPS",
        option_type="Put",
        strike=80.0,
        expiration_date="2027-06-17",
        cost_basis=3.37,
    )
    console.print(
        f"  • Basis Long Put: [bold yellow]{basis_long.symbol} ${basis_long.strike:.2f}P[/bold yellow] "
        f"Exp: [bold white]{basis_long.expiration_date}[/bold white] | Cost Basis: [bold green]${basis_long.cost_basis:.2f}[/bold green] / share ($337.00/contract)"
    )

    # Initialize Strategy Analyzer
    analyzer = DiagonalSpreadAnalyzer(
        basis_long=basis_long,
        target_yield=args.target_yield,
        risk_free_rate=0.045,
    )

    console.print(f"\n[bold blue]Step 3: Filtering Candidate Short Puts[/bold blue]")
    console.print(f"  • Filter 1: [bold]Option Type == 'Put'[/bold]")
    console.print(f"  • Filter 2: [bold]OTM Only (Strike < Spot Price ${spot_price:.2f})[/bold]")
    console.print(f"  • Filter 3: [bold]Delta in range [{args.min_delta:.2f}, {args.max_delta:.2f}][/bold]")
    console.print(f"  • Filter 4: [bold]Mid Price > 0[/bold]")

    results_df, diag_df = analyzer.analyze_dataset(
        df_raw, spot_price=spot_price, min_delta=args.min_delta, max_delta=args.max_delta
    )

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save diagnostic breakdown
    diag_csv_path = output_dir / "put_filtering_diagnostics.csv"
    diag_df.to_csv(diag_csv_path, index=False)

    num_selected = len(results_df)
    num_excluded = len(diag_df) - num_selected
    console.print(f"\n  → Filter Results: [bold green]{num_selected} puts selected[/bold green], [bold red]{num_excluded} puts excluded[/bold red].")

    # Display Diagnostic Breakdown Table of Puts near threshold
    diag_table = Table(
        title="Put Options Scan & Filtering Breakdown (Strikes near Spot / Relevant Deltas)",
        title_style="bold yellow",
        header_style="bold cyan",
        show_lines=True,
    )
    diag_table.add_column("Exp Date", justify="center")
    diag_table.add_column("Strike", justify="right")
    diag_table.add_column("Delta", justify="right")
    diag_table.add_column("|Delta|", justify="right")
    diag_table.add_column("Mid ($)", justify="right")
    diag_table.add_column("Status", justify="center")
    diag_table.add_column("Filter Decision / Reason", justify="left")

    # Focus table on relevant strikes (e.g. Strike >= 75 or Delta >= 0.05)
    relevant_diag = diag_df[(diag_df["strike"] >= 75.0) | (diag_df["abs_delta"] >= 0.05)].copy()
    if args.show_all_puts:
        relevant_diag = diag_df.copy()

    for _, row in relevant_diag.iterrows():
        status_str = "[bold green]SELECTED[/bold green]" if row["status"] == "SELECTED" else "[red]EXCLUDED[/red]"
        reason_color = "white" if row["status"] == "SELECTED" else "dim red"
        diag_table.add_row(
            str(row["expiration_date"]),
            f"${row['strike']:.2f}",
            f"{row['delta']:+.4f}",
            f"{row['abs_delta']:.4f}",
            f"${row['mid']:.2f}",
            status_str,
            f"[{reason_color}]{row['reason']}[/{reason_color}]",
        )

    console.print(diag_table)

    if results_df.empty:
        console.print(f"[yellow]No short put candidates matched the filter criteria.[/yellow]")
        return

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

    # Display Strategy Ranking Table
    table = Table(
        title="Final Selected Short Puts for Diagonal Spread (Sorted by Delta)",
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

    console.print(f"\n[bold blue]Step 4: Selected Candidates Ranked Table[/bold blue]")
    console.print(table)
    console.print(f"\n[bold green]✓[/bold green] Analysis complete! Results written to:")
    console.print(f"  • [cyan]{csv_path}[/cyan]")
    console.print(f"  • [cyan]{md_path}[/cyan]")
    console.print(f"  • [cyan]{diag_csv_path}[/cyan] (full diagnostics log for all {total_puts} puts)\n")


if __name__ == "__main__":
    main()
