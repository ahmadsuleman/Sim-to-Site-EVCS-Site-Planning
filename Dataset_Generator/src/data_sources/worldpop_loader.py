from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd


class WorldPopLoader:
    """Local-file adapter for WorldPop-derived point or raster samples.

    Expected CSV columns for lightweight use: lat, lon, population_density.
    Full raster sampling can be added with rasterio in deployments that install it.
    """
    def load_csv(self, path: Path) -> Optional[pd.DataFrame]:
        if path.exists():
            return pd.read_csv(path)
        return None
