from __future__ import annotations

from pathlib import Path
from typing import Dict
import pandas as pd


class CSVExporter:
    def __init__(self, csv_dir: Path):
        self.csv_dir = csv_dir
        self.csv_dir.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, df: pd.DataFrame) -> Path:
        path = self.csv_dir / name
        df.to_csv(path, index=False)
        return path

    def write_many(self, frames: Dict[str, pd.DataFrame]) -> None:
        for name, df in frames.items():
            self.write(name, df)
