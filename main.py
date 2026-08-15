"""Main entry point for Short Options Profit & Diagonal Spread Selection Analyzer with Multi-Asset Support."""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import pandas as pd
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

# Add src to python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from options_analyzer.loader import DataLoader
from options_analyzer.analyzer import DiagonalSpreadAnalyzer, LongOptionPosition, StrategyRules


def process_single_underlying(
    console: Console,
    rules: StrategyRules,
    basis_pos: LongOptionPosition,
    df_raw: pd.DataFrame,
    output_dir: Path,
    show_all_puts: bool = False,
) -> Optional[pd.DataFrame]:
    """Process analysis for a single underlying asset and write reports."""
    symbol = basis_pos.symbol.upper()
    console.print(f"\n[bold blue]─── Analyzing Underlying: [bold yellow]{symbol}[/bold yellow] ───[/bold blue]")

    # Filter raw dataframe for this symbol if present
    if "symbol" in df_raw.columns:
        df_symbol = df_raw[df_raw["symbol"].str.upper() == symbol].copy()
    else:
        df_symbol = df_raw.copy()

    if df_symbol.empty:
        console.print(f"[yellow]No data files or rows found for symbol [bold]{symbol}[/bold] in source/ folder.[/yellow]")
        return None

    # Estimate Spot Price for this symbol
    spot_price = DataLoader.estimate_spot_price(df_symbol)
    console.print(f"  • Estimated Spot Price ({symbol}): [bold green]${spot_price:.2f}[/bold green]")
    console.print(
        f"  • Basis Long Put: [bold yellow]{basis_pos.symbol} ${basis_pos.strike:.2f}P[/bold yellow] "
        f"Exp: [bold white]{basis_pos.expiration_date}[/bold white] | Cost Basis: [bold green]${basis_pos.cost_basis:.2f}[/bold green] / share (${basis_pos.cost_basis * 100:.2f}/contract)"
    )

    analyzer = DiagonalSpreadAnalyzer(basis_long=basis_pos, rules=rules)
    results_df, diag_df = analyzer.analyze_dataset(df_symbol, spot_price=spot_price, symbol=symbol)

    # Save per-symbol diagnostics
    diag_csv_path = output_dir / f"put_filtering_diagnostics_{symbol}.csv"
    diag_df.to_csv(diag_csv_path, index=False)

    num_selected = len(results_df)
    num_excluded = len(diag_df) - num_selected
    console.print(f"  → Filter Results: [bold green]{num_selected} puts selected[/bold green], [bold red]{num_excluded} puts excluded[/bold red].")

    # Diagnostics breakdown table
    diag_table = Table(
        title=f"Put Options Scan Breakdown for {symbol} (Strikes near Delta Bounds)",
        title_style="bold yellow",
        header_style="bold cyan",
        show_lines=True,
    )
    diag_table.add_column("Symbol", justify="center")
    diag_table.add_column("Exp Date", justify="center")
    diag_table.add_column("Strike", justify="right")
    diag_table.add_column("Delta", justify="right")
    diag_table.add_column("|Delta|", justify="right")
    diag_table.add_column("Mid ($)", justify="right")
    diag_table.add_column("Status", justify="center")
    diag_table.add_column("Filter Decision / Reason", justify="left")

    relevant_diag = diag_df[(diag_df["strike"] >= 75.0) & (diag_df["strike"] <= 125.0)].copy()
    if show_all_puts or relevant_diag.empty:
        relevant_diag = diag_df.copy()

    for _, row in relevant_diag.iterrows():
        status_str = "[bold green]SELECTED[/bold green]" if row["status"] == "SELECTED" else "[red]EXCLUDED[/red]"
        reason_color = "white" if row["status"] == "SELECTED" else "dim red"
        diag_table.add_row(
            str(row["symbol"]),
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
        console.print(f"[yellow]No short put candidates matched filter criteria for {symbol}.[/yellow]")
        return None

    display_cols = [
        "symbol",
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

    # Save per-symbol outputs
    sym_csv = output_dir / f"diagonal_spread_analysis_{symbol}.csv"
    sym_md = output_dir / f"diagonal_spread_analysis_{symbol}.md"
    export_df.to_csv(sym_csv, index=False)

    md_content = f"""# Short Put Options Selection & Diagonal Spread Analysis ({symbol})

**Underlying**: {symbol} ($ {spot_price:.2f})  
**Basis Long Put**: {basis_pos.symbol} ${basis_pos.strike:.2f}P Exp: {basis_pos.expiration_date} (Cost: ${basis_pos.cost_basis:.2f} / share)  
**Strategy**: Diagonal Put Spread with {rules.target_yield * 100:.0f}% Extrinsic Profit Target (20% residual extrinsic decay)  
**Delta Filter**: [{rules.min_delta:.2f}, {rules.max_delta:.2f}]  
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

    with open(sym_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Rich Summary Table
    table = Table(
        title=f"Final Selected Short Puts for {symbol} Diagonal Spread (Sorted by Delta)",
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
    console.print(f"  • Wrote report: [cyan]{sym_csv}[/cyan] and [cyan]{sym_md}[/cyan]")

    return export_df


def main():
    parser = argparse.ArgumentParser(description="Analyze diagonal spread short put candidates with multi-underlying support.")
    parser.add_argument("--config", default="rules.yaml", help="Path to YAML rules configuration file (default: rules.yaml)")
    parser.add_argument("--source", default="source", help="Directory containing downloaded options files (default: source)")
    parser.add_argument("--symbol", default=None, help="Target underlying symbol (e.g. UPS, XOM, or ALL). If omitted, an interactive prompt is shown.")
    parser.add_argument("--min-delta", type=float, default=None, help="Override minimum absolute delta (e.g. 0.10)")
    parser.add_argument("--max-delta", type=float, default=None, help="Override maximum absolute delta (e.g. 0.55)")
    parser.add_argument("--show-all-puts", action="store_true", help="Show full diagnostics of all scanned put options")
    args = parser.parse_args()

    console = Console()
    console.print("\n[bold cyan]═══ Short Options Profit & Diagonal Spread Selection ═══[/bold cyan]\n")

    # Load configuration from YAML rules file
    rules = StrategyRules.from_yaml(args.config)
    if args.min_delta is not None:
        rules.min_delta = args.min_delta
    if args.max_delta is not None:
        rules.max_delta = args.max_delta

    available_symbols = rules.list_symbols()
    console.print(f"[bold blue]Configuration loaded from:[/bold blue] [cyan]{args.config}[/cyan]")
    console.print(f"  • Configured Underlyings: [bold yellow]{', '.join(available_symbols)}[/bold yellow]")
    console.print(f"  • Delta Range: [bold green]{rules.min_delta:.2f} - {rules.max_delta:.2f}[/bold green]")
    console.print(f"  • Require Strike < Spot: [bold magenta]{rules.require_strike_less_than_spot}[/bold magenta]")
    console.print(f"  • Profit Target: [bold green]{rules.target_yield * 100:.0f}%[/bold green] of extrinsic value\n")

    loader = DataLoader(args.source)
    files = loader.list_files()

    if not files:
        console.print(f"[yellow]No data files found in [bold]{args.source}/[/bold] folder.[/yellow]")
        console.print(f"Place your options CSV/Parquet files into [bold]{args.source}/[/bold] to begin analysis.")
        return

    console.print(f"[bold blue]Available Data Files in {args.source}/:[/bold blue]")
    for f in files:
        console.print(f"  • [cyan]{f.name}[/cyan] ({f.stat().st_size / 1024:.1f} KB)")

    df_raw = loader.load_all()
    if df_raw.empty:
        console.print("[red]Could not load data from files in source/.[/red]")
        return

    # Select underlying symbol
    target_symbol = args.symbol
    if not target_symbol:
        # Prompt user interactively
        choices = available_symbols + ["ALL"]
        console.print("\n[bold cyan]Select Underlying to Analyze:[/bold cyan]")
        for i, sym in enumerate(choices, 1):
            if sym == "ALL":
                console.print(f"  [{i}] [bold green]ALL[/bold green] (Process all configured underlyings)")
            else:
                pos = rules.get_basis_position(sym)
                console.print(f"  [{i}] [bold yellow]{sym}[/bold yellow] (Basis Long: {pos.strike:.2f}P {pos.expiration_date} @ ${pos.cost_basis:.2f})")

        choice_input = Prompt.ask(
            "\nEnter choice number or symbol name",
            choices=[str(i) for i in range(1, len(choices) + 1)] + [s.lower() for s in choices] + choices,
            default="1",
        )

        if choice_input.isdigit() and 1 <= int(choice_input) <= len(choices):
            target_symbol = choices[int(choice_input) - 1]
        else:
            target_symbol = choice_input.upper()

    target_symbol = target_symbol.upper().strip()
    console.print(f"\n[bold green]Proceeding with selection:[/bold green] [bold white]{target_symbol}[/bold white]")

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    symbols_to_process = available_symbols if target_symbol == "ALL" else [target_symbol]
    all_exports: List[pd.DataFrame] = []

    for sym in symbols_to_process:
        pos = rules.get_basis_position(sym)
        if not pos:
            console.print(f"[red]Error: Symbol '{sym}' is not configured in {args.config}.[/red]")
            continue

        res_df = process_single_underlying(
            console=console,
            rules=rules,
            basis_pos=pos,
            df_raw=df_raw,
            output_dir=output_dir,
            show_all_puts=args.show_all_puts,
        )
        if res_df is not None and not res_df.empty:
            all_exports.append(res_df)

    if all_exports:
        combined_df = pd.concat(all_exports, ignore_index=True)
        combined_csv = output_dir / "diagonal_spread_analysis.csv"
        combined_md = output_dir / "diagonal_spread_analysis.md"
        combined_df.to_csv(combined_csv, index=False)

        # Write combined markdown report
        md_content = f"""# Multi-Asset Short Options Selection & Diagonal Spread Analysis

**Strategy**: Diagonal Put Spread with {rules.target_yield * 100:.0f}% Extrinsic Profit Target  
**Delta Filter**: [{rules.min_delta:.2f}, {rules.max_delta:.2f}]  
**Valuation Rule**: Always use Medium price for Bid/Ask: $\\text{{Mid}} = \\frac{{\\text{{Bid}} + \\text{{Ask}}}}{{2}}$  

## Combined Candidate Short Puts

| Symbol | Delta | Short Put Identifier | Daily Rel Profit (%) | Days to Target | Profit ($) | Strike ($) | Expiration | DTE | Mid Price ($) | Spread Risk ($) | Daily Profit ($) | IV (%) |
|--------|-------|----------------------|----------------------|----------------|------------|------------|-----|---------------|-----------------|------------------|--------|
"""
        for _, row in combined_df.iterrows():
            md_content += (
                f"| **{row['symbol']}** | {row['delta']:+.4f} | `{row['short_put_index']}` | **{row['daily_relative_profit']:.3f}%** | "
                f"{row['days_to_target']:.1f} | ${row['profit_usd']:.2f} | ${row['strike']:.2f} | "
                f"{row['expiration_date']} | {row['dte']} | ${row['mid_price']:.2f} | "
                f"${row['spread_risk_usd']:.2f} | ${row['daily_profit_usd']:.2f} | {row['iv_pct']:.1f}% |\n"
            )

        with open(combined_md, "w", encoding="utf-8") as f:
            f.write(md_content)

        console.print(f"\n[bold green]✓ Combined results updated in:[/bold green] [cyan]{combined_csv}[/cyan] and [cyan]{combined_md}[/cyan]\n")


if __name__ == "__main__":
    main()
