from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd


class OpenChargeMapLoader:
    """Local-file adapter for OpenChargeMap extracts.

    Expected CSV columns: lat, lon, power_kw. API download can be added where an API key and internet are available.
    """
    def load_csv(self, path: Path) -> Optional[pd.DataFrame]:
        if path.exists():
            return pd.read_csv(path)
        return None
