"""Data loader module for discovering, loading, cleaning, and extracting metadata from options datasets."""

import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class DataLoader:
    """Discovers, parses, cleans, and structures options datasets from the source directory."""

    def __init__(self, source_dir: Union[str, Path] = "source"):
        self.source_dir = Path(source_dir)

    def list_files(self) -> List[Path]:
        """List all supported data files located in the source directory."""
        if not self.source_dir.exists():
            return []

        supported_extensions = {".csv", ".parquet", ".pq", ".xlsx", ".xls", ".json", ".tsv"}
        files = [
            p for p in self.source_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in supported_extensions
        ]
        return sorted(files)

    @staticmethod
    def _clean_numeric_col(series: pd.Series) -> pd.Series:
        """Clean currency, commas, percentages, and 'unch' entries into numeric floats."""
        if pd.api.types.is_numeric_dtype(series):
            return series

        cleaned = (
            series.astype(str)
            .str.strip()
            .replace(["unch", "N/A", "nan", "--", ""], np.nan)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace("+", "", regex=False)
        )
        return pd.to_numeric(cleaned, errors="coerce")

    @staticmethod
    def parse_filename_metadata(filename: str) -> dict:
        """Extract metadata (symbol, expiration, quote date) from typical options filenames."""
        metadata = {}
        exp_match = re.search(r"exp-(\d{4}-\d{2}-\d{2})", filename, re.IGNORECASE)
        if exp_match:
            metadata["expiration_date"] = exp_match.group(1)

        symbol_match = re.match(r"^([a-zA-Z0-9]+)-options", filename)
        if symbol_match:
            metadata["symbol"] = symbol_match.group(1).upper()

        quote_date_match = re.search(r"(\d{2}-\d{2}-\d{4})\.[a-zA-Z0-9]+$", filename)
        if quote_date_match:
            # Parse MM-DD-YYYY to YYYY-MM-DD
            qd_raw = quote_date_match.group(1)
            try:
                dt = datetime.strptime(qd_raw, "%m-%d-%Y")
                metadata["quote_date"] = dt.strftime("%Y-%m-%d")
            except ValueError:
                metadata["quote_date"] = qd_raw

        return metadata

    @staticmethod
    def estimate_spot_price(df: pd.DataFrame) -> float:
        """Estimate the underlying spot price using strike and moneyness or closest ATM strikes."""
        if "Strike" in df.columns and "Moneyness" in df.columns:
            # Moneyness is (S - K)/S * 100 for Calls or (K - S)/S * 100 for Puts
            # Or Moneyness is (S - K)/K. Let's inspect rows with small moneyness
            calls = df[df["Type"].str.lower() == "call"] if "Type" in df.columns else df
            if not calls.empty and "Moneyness" in calls.columns:
                valid = calls.dropna(subset=["Strike", "Moneyness"]).copy()
                if not valid.empty:
                    # Sort by absolute moneyness
                    valid["abs_m"] = valid["Moneyness"].abs()
                    atm_rows = valid.sort_values("abs_m").head(3)
                    # S = K / (1 - m/100) or S = K * (1 + m/100)
                    estimates = []
                    for _, row in atm_rows.iterrows():
                        k = row["Strike"]
                        m = row["Moneyness"] / 100.0
                        # Try S = K / (1 - m)
                        s1 = k / (1.0 - m) if abs(1.0 - m) > 1e-4 else k
                        estimates.append(s1)
                    if estimates:
                        return float(np.round(np.median(estimates), 2))

        # Fallback to strike with delta closest to 0.50 for calls
        if "Delta" in df.columns and "Strike" in df.columns:
            calls = df[df["Type"].str.lower() == "call"] if "Type" in df.columns else df
            if not calls.empty:
                closest_atm = calls.iloc[(calls["Delta"] - 0.5).abs().argsort()[:1]]
                if not closest_atm.empty:
                    return float(closest_atm["Strike"].values[0])

        return 100.0

    def load_file(self, file_path: Union[str, Path], nrows: Optional[int] = None, clean: bool = True) -> pd.DataFrame:
        """Load a single data file into a pandas DataFrame and extract all fields."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix in {".csv", ".tsv"}:
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(path, sep=sep, nrows=nrows)
        elif suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
            df = df.head(nrows) if nrows else df
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(path, nrows=nrows)
        elif suffix == ".json":
            df = pd.read_json(path)
            df = df.head(nrows) if nrows else df
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        meta = self.parse_filename_metadata(path.name)
        if "symbol" in meta and "symbol" not in df.columns:
            df["symbol"] = meta["symbol"]
        if "expiration_date" in meta and "expiration_date" not in df.columns:
            df["expiration_date"] = meta["expiration_date"]
        if "quote_date" in meta and "quote_date" not in df.columns:
            df["quote_date"] = meta["quote_date"]

        if clean:
            df = self.clean_dataframe(df)

        # Compute DTE if dates are present
        if "expiration_date" in df.columns and "quote_date" in df.columns:
            try:
                exp_dt = pd.to_datetime(df["expiration_date"])
                quote_dt = pd.to_datetime(df["quote_date"])
                df["dte"] = (exp_dt - quote_dt).dt.days
            except Exception:
                df["dte"] = 30

        return df

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and clean data types for options data."""
        df = df.copy()

        # Clean numeric fields if present
        for col in ["Strike", "Bid", "Mid", "Ask", "Latest", "Change", "%Change", "Volume", "Open Int", "OI Chg", "Delta", "Moneyness"]:
            if col in df.columns:
                df[col] = self._clean_numeric_col(df[col])

        # If IV has % sign, convert percentage to decimal float (e.g. 26.55% -> 0.2655 or keep as decimal)
        if "IV" in df.columns:
            df["IV"] = self._clean_numeric_col(df["IV"]) / 100.0

        # Always enforce Mid = (Bid + Ask) / 2 rule if Bid & Ask exist
        if "Bid" in df.columns and "Ask" in df.columns:
            computed_mid = (df["Bid"] + df["Ask"]) / 2.0
            # If Mid column is missing or differs, enforce medium price
            df["Mid"] = computed_mid.round(4)

        return df

    def load_all(self, pattern: Optional[str] = None, clean: bool = True) -> pd.DataFrame:
        """Load and concatenate all matching files in the source directory."""
        files = self.list_files()
        if pattern:
            files = [f for f in files if f.match(pattern)]

        if not files:
            return pd.DataFrame()

        dfs = [self.load_file(f, clean=clean) for f in files]
        return pd.concat(dfs, ignore_index=True)
