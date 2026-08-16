"""Main entry point for Short Options Profit Analyzer supporting:
1. Diagonal Put Spreads
2. Cash Protected Puts (Cash Secured Puts)
3. Vertical Put Spreads (Credit Spreads)
"""

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
from options_analyzer.analyzer import ShortOptionsAnalyzer, LongOptionPosition, StrategyRules, StrategyType


def process_single_underlying(
    console: Console,
    rules: StrategyRules,
    symbol: str,
    df_raw: pd.DataFrame,
    output_dir: Path,
    strat_type: StrategyType,
    basis_pos: Optional[LongOptionPosition] = None,
    show_all_puts: bool = False,
) -> Optional[pd.DataFrame]:
    """Process analysis for a single underlying asset and strategy type."""
    sym = symbol.upper()

    if strat_type == StrategyType.DIAGONAL_SPREAD:
        strategy_title = "Diagonal Put Spread"
        prefix = "diagonal_spread_analysis"
    elif strat_type == StrategyType.VERTICAL_SPREAD:
        strategy_title = "Vertical Put Spread"
        prefix = "vertical_spread_analysis"
    else:
        strategy_title = "Cash Protected Put"
        prefix = "cash_protected_put_analysis"

    console.print(f"\n[bold blue]─── Analyzing Underlying: [bold yellow]{sym}[/bold yellow] ({strategy_title}) ───[/bold blue]")

    # Filter raw dataframe for this symbol if present
    if "symbol" in df_raw.columns:
        df_symbol = df_raw[df_raw["symbol"].str.upper() == sym].copy()
    else:
        df_symbol = df_raw.copy()

    if df_symbol.empty:
        console.print(f"[yellow]No data files or rows found for symbol [bold]{sym}[/bold] in source/ folder.[/yellow]")
        return None

    # Estimate Spot Price for this symbol
    spot_price = DataLoader.estimate_spot_price(df_symbol)
    console.print(f"  • Estimated Spot Price ({sym}): [bold green]${spot_price:.2f}[/bold green]")
    
    if strat_type == StrategyType.DIAGONAL_SPREAD and basis_pos:
        console.print(
            f"  • Strategy: [bold cyan]Diagonal Put Spread[/bold cyan] | Basis Long: [bold yellow]{basis_pos.symbol} ${basis_pos.strike:.2f}P[/bold yellow] "
            f"Exp: [bold white]{basis_pos.expiration_date}[/bold white] | Cost Basis: [bold green]${basis_pos.cost_basis:.2f}[/bold green] / share (${basis_pos.cost_basis * 100:.2f}/contract)"
        )
    elif strat_type == StrategyType.VERTICAL_SPREAD:
        console.print(
            f"  • Strategy: [bold magenta]Vertical Put Spread[/bold magenta] (Protecting Long Put in same expiration: $\\Delta_{{\\text{{target}}}} \\approx |\\Delta_{{\\text{{short}}}}| - {rules.vertical_target_delta_offset:.2f}$, Min $\\Delta \\ge {rules.vertical_min_long_delta:.2f}$)"
        )
    else:
        console.print(
            f"  • Strategy: [bold cyan]Cash Protected Put[/bold cyan] (No protecting long position; Max Risk = Strike - Mid)"
        )

    analyzer = ShortOptionsAnalyzer(
        basis_long=basis_pos,
        rules=rules,
        strategy_type=strat_type,
    )
    results_df, diag_df = analyzer.analyze_dataset(df_symbol, spot_price=spot_price, symbol=sym)

    num_selected = len(results_df)
    num_excluded = len(diag_df) - num_selected
    console.print(f"  → Filter Results: [bold green]{num_selected} puts selected[/bold green], [bold red]{num_excluded} puts excluded[/bold red].")

    # Diagnostics breakdown table
    diag_table = Table(
        title=f"Put Options Scan Breakdown for {sym} (Strikes near Delta Bounds [{rules.min_delta:.2f}, {rules.max_delta:.2f}])",
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

    if not diag_df.empty:
        min_p_strike = diag_df["strike"].min()
        max_p_strike = diag_df["strike"].max()
        relevant_diag = diag_df.copy()
        if not show_all_puts and (max_p_strike - min_p_strike > 50):
            relevant_diag = diag_df[(diag_df["strike"] >= spot_price * 0.75) & (diag_df["strike"] <= spot_price * 1.25)].copy()
            if relevant_diag.empty:
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
        console.print(f"[yellow]No candidate options matched filter criteria for {sym} {strategy_title}.[/yellow]")
        return None

    # Column ordering
    display_cols = [
        "symbol",
        "strategy_type",
        "delta",
        "short_put_index",
        "expected_daily_relative_profit",
        "daily_relative_profit",
        "days_to_target",
        "profit_usd",
        "max_risk_usd",
        "target_profit_usd",
        "target_yield_pct",
        "p_win_pct",
        "strike",
        "expiration_date",
        "dte",
        "mid_price",
        "daily_profit_usd",
        "delta_efficiency",
        "iv_pct",
    ]
    if strat_type == StrategyType.VERTICAL_SPREAD:
        display_cols.extend(["long_strike", "long_mid", "long_delta"])

    export_df = results_df[[c for c in display_cols if c in results_df.columns]].copy()

    # Save per-symbol outputs
    sym_csv = output_dir / f"{prefix}_{sym}.csv"
    sym_md = output_dir / f"{prefix}_{sym}.md"
    export_df.to_csv(sym_csv, index=False)

    if strat_type == StrategyType.DIAGONAL_SPREAD and basis_pos:
        long_info = (
            f"**Basis Long Put**: {basis_pos.symbol} ${basis_pos.strike:.2f}P Exp: {basis_pos.expiration_date} (Cost: ${basis_pos.cost_basis:.2f} / share)  \n"
            f"**Max Risk Formula**: $(K_{{\\text{{short}}}} - K_{{\\text{{long}}}}) + (\\text{{Cost}}_{{\\text{{long}}}} - \\text{{Mid}}_{{\\text{{short}}}})$  "
        )
    elif strat_type == StrategyType.VERTICAL_SPREAD:
        long_info = (
            f"**Strategy Type**: Vertical Put Spread (Credit Spread)  \n"
            f"**Long Put Selection**: Same expiration, $\\Delta \\approx |\\Delta_{{\\text{{short}}}}| - {rules.vertical_target_delta_offset:.2f}$, Min $\\Delta \\ge {rules.vertical_min_long_delta:.2f}$  \n"
            f"**Max Risk Formula**: $(K_{{\\text{{short}}}} - K_{{\\text{{long}}}}) - (\\text{{Mid}}_{{\\text{{short}}}} - \\text{{Mid}}_{{\\text{{long}}}})$ (Strike Width minus Net Credit)  "
        )
    else:
        long_info = "**Position Type**: Cash Protected Put (No protecting long position)  \n**Max Risk Formula**: $K_{{\\text{{short}}}} - \\text{{Mid}}_{{\\text{{short}}}}$ (Strike minus cost of PUT)  "

    target_yield_val = analyzer.target_yield

    md_content = f"""# Short Put Options Selection & {strategy_title} ({sym})

**Underlying**: {sym} ($ {spot_price:.2f})  
**Strategy**: {strategy_title} with {target_yield_val * 100:.0f}% Extrinsic/Credit Profit Target  
{long_info}
**Delta Filter**: [{rules.min_delta:.2f}, {rules.max_delta:.2f}]  
**Valuation Rule**: Always use Medium price for Bid/Ask: $\\text{{Mid}} = \\frac{{\\text{{Bid}} + \\text{{Ask}}}}{{2}}$  

## Candidate Short Puts (Sorted by Delta)

| Delta | Short Put Identifier | Expected Daily Rel Profit (%) | Daily Rel Profit (%) | Days to Target | Profit ($) | Max Risk ($) | Target Profit ($) | Target Yield (%) | Win Prob | Strike ($) | Expiration | DTE | Mid Price ($) | Daily Profit ($) | IV (%) |
|-------|----------------------|-------------------------------|----------------------|----------------|------------|--------------|-------------------|------------------|----------|------------|------------|-----|---------------|------------------|--------|
"""
    for _, row in export_df.iterrows():
        md_content += (
            f"| {row['delta']:+.4f} | `{row['short_put_index']}` | **{row['expected_daily_relative_profit']:.3f}%** | "
            f"{row['daily_relative_profit']:.3f}% | {row['days_to_target']:.1f} | ${row['profit_usd']:.2f} | "
            f"${row['max_risk_usd']:.2f} | ${row['target_profit_usd']:.2f} | **{row['target_yield_pct']:.2f}%** | "
            f"{row['p_win_pct']:.1f}% | ${row['strike']:.2f} | {row['expiration_date']} | {row['dte']} | "
            f"${row['mid_price']:.2f} | ${row['daily_profit_usd']:.2f} | {row['iv_pct']:.1f}% |\n"
        )

    with open(sym_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Rich Summary Table
    table = Table(
        title=f"Final Selected Short Puts for {sym} {strategy_title} (Sorted by Delta)",
        title_style="bold magenta",
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Delta", justify="right", style="cyan")
    table.add_column("Spread / Put Index", justify="left", style="bold white")
    table.add_column("Expected Daily Rel", justify="right", style="bold green")
    table.add_column("Daily Rel", justify="right", style="green")
    table.add_column("Days to Target", justify="right", style="yellow")
    table.add_column("Profit ($)", justify="right", style="white")
    table.add_column("Max Risk ($)", justify="right", style="red")
    table.add_column("Target Profit ($)", justify="right", style="bold green")
    table.add_column("Target Yield (%)", justify="right", style="bold yellow")
    table.add_column("Win Prob", justify="right", style="magenta")
    table.add_column("Strike", justify="right")
    table.add_column("Exp Date", justify="center")
    table.add_column("DTE", justify="right")
    table.add_column("Mid ($)", justify="right")

    for _, row in export_df.iterrows():
        table.add_row(
            f"{row['delta']:+.4f}",
            str(row["short_put_index"]),
            f"{row['expected_daily_relative_profit']:.3f}%",
            f"{row['daily_relative_profit']:.3f}%",
            f"{row['days_to_target']:.1f}",
            f"${row['profit_usd']:.2f}",
            f"${row['max_risk_usd']:.2f}",
            f"${row['target_profit_usd']:.2f}",
            f"{row['target_yield_pct']:.2f}%",
            f"{row['p_win_pct']:.1f}%",
            f"${row['strike']:.2f}",
            str(row["expiration_date"]),
            str(row["dte"]),
            f"${row['mid_price']:.2f}",
        )

    console.print(table)
    console.print(f"  • Wrote report: [cyan]{sym_csv}[/cyan] and [cyan]{sym_md}[/cyan]")

    return export_df


def main():
    parser = argparse.ArgumentParser(description="Analyze Diagonal Put Spreads, Cash Protected Puts, and Vertical Put Spreads.")
    parser.add_argument("--config", default="rules.yaml", help="Path to YAML rules configuration file (default: rules.yaml)")
    parser.add_argument("--positions", default="basis_long_positions.csv", help="Path to CSV basis long positions file (default: basis_long_positions.csv)")
    parser.add_argument("--source", default="source", help="Directory containing downloaded options files (default: source)")
    parser.add_argument("--symbol", default=None, help="Target underlying symbol (e.g. AAPL, PLTR, UPS, XOM, or ALL). If omitted, an interactive prompt is shown.")
    parser.add_argument("--strategy", default="all_applicable", choices=["all_applicable", "diagonal_spread", "cash_secured_put", "vertical_spread", "all_strategies"], help="Strategy to run (default: all_applicable)")
    parser.add_argument("--min-delta", type=float, default=None, help="Override minimum absolute delta (e.g. 0.15)")
    parser.add_argument("--max-delta", type=float, default=None, help="Override maximum absolute delta (e.g. 0.55)")
    parser.add_argument("--show-all-puts", action="store_true", help="Show full diagnostics of all scanned put options")
    args = parser.parse_args()

    console = Console()
    console.print("\n[bold cyan]═══ Short Options Profit & Multi-Strategy Analyzer ═══[/bold cyan]\n")

    # Load configuration from YAML rules file and CSV positions file
    rules = StrategyRules.from_yaml(args.config, positions_file=args.positions)
    if args.min_delta is not None:
        rules.min_delta = args.min_delta
    if args.max_delta is not None:
        rules.max_delta = args.max_delta

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

    # Discover all symbols from loaded data and long positions
    raw_symbols = sorted(list(df_raw["symbol"].dropna().unique())) if "symbol" in df_raw.columns else []
    configured_symbols = rules.list_symbols()
    all_symbols = sorted(list(set(raw_symbols + configured_symbols)))

    console.print(f"\n[bold blue]Configuration loaded:[/bold blue]")
    console.print(f"  • Discovered Underlyings: [bold yellow]{', '.join(all_symbols)}[/bold yellow]")
    console.print(f"  • Delta Range: [bold green]{rules.min_delta:.2f} - {rules.max_delta:.2f}[/bold green]")
    console.print(f"  • Profit Targets: [bold green]Diagonal 80% | Cash Protected 50% | Vertical 50%[/bold green]")
    console.print(f"  • Vertical Spread Long Offset: [bold cyan]-{rules.vertical_target_delta_offset:.2f} Delta (Min {rules.vertical_min_long_delta:.2f})[/bold cyan]\n")

    # Select underlying symbol
    target_symbol = args.symbol
    if not target_symbol:
        choices = all_symbols + ["ALL"]
        console.print("[bold cyan]Select Underlying to Analyze:[/bold cyan]")
        for i, sym in enumerate(choices, 1):
            if sym == "ALL":
                console.print(f"  [{i}] [bold green]ALL[/bold green] (Process all discovered underlyings)")
            else:
                pos = rules.get_basis_position(sym)
                if pos:
                    console.print(f"  [{i}] [bold yellow]{sym}[/bold yellow] (Basis Long: {pos.strike:.2f}P {pos.expiration_date} @ ${pos.cost_basis:.2f})")
                else:
                    console.print(f"  [{i}] [bold yellow]{sym}[/bold yellow] (Available for Cash Protected Puts & Vertical Spreads)")

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

    symbols_to_process = all_symbols if target_symbol == "ALL" else [target_symbol]

    for sym in symbols_to_process:
        pos = rules.get_basis_position(sym)
        
        # Decide which strategies to run for this symbol
        strategies_to_run = []
        if args.strategy == "diagonal_spread":
            if pos:
                strategies_to_run.append(StrategyType.DIAGONAL_SPREAD)
            else:
                console.print(f"[yellow]Skipping Diagonal Spread for {sym} (no basis long configured in {args.positions})[/yellow]")
        elif args.strategy == "cash_secured_put":
            strategies_to_run.append(StrategyType.CASH_PROTECTED_PUT)
        elif args.strategy == "vertical_spread":
            strategies_to_run.append(StrategyType.VERTICAL_SPREAD)
        elif args.strategy == "all_strategies":
            if pos:
                strategies_to_run.append(StrategyType.DIAGONAL_SPREAD)
            strategies_to_run.append(StrategyType.CASH_PROTECTED_PUT)
            strategies_to_run.append(StrategyType.VERTICAL_SPREAD)
        else:
            # Default "all_applicable"
            if pos:
                strategies_to_run.append(StrategyType.DIAGONAL_SPREAD)
            strategies_to_run.append(StrategyType.CASH_PROTECTED_PUT)
            strategies_to_run.append(StrategyType.VERTICAL_SPREAD)

        for strat in strategies_to_run:
            process_single_underlying(
                console=console,
                rules=rules,
                symbol=sym,
                df_raw=df_raw,
                output_dir=output_dir,
                strat_type=strat,
                basis_pos=pos if strat == StrategyType.DIAGONAL_SPREAD else None,
                show_all_puts=args.show_all_puts,
            )

    console.print(f"\n[bold green]✓ All analysis reports generated in [cyan]{output_dir}/[/cyan][/bold green]\n")


if __name__ == "__main__":
    main()
