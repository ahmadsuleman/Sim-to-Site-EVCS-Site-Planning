from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def ensure_dirs(root: str | Path) -> Dict[str, Path]:
    root = Path(root)
    dirs = {
        'root': root,
        'metadata': root / 'metadata',
        'cache': root / 'cache',
        'oracle': root / 'oracle',
        'oracle_selected_sites': root / 'oracle' / 'oracle_selected_sites',
        'oracle_assignments': root / 'oracle' / 'oracle_assignments',
        'labels': root / 'labels',
        'models': root / 'models',
        'predictions': root / 'predictions',
        'rankings': root / 'rankings',
        'transfer': root / 'transfer',
        'sensitivity': root / 'sensitivity',
        'tables': root / 'tables',
        'figures': root / 'figures',
        'reports': root / 'reports',
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
