from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd


class LocalFileLoader:
    def load_csv(self, path: Path) -> Optional[pd.DataFrame]:
        return pd.read_csv(path) if path.exists() else None
