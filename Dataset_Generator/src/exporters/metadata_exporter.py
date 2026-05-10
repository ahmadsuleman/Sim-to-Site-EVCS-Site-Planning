from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from src.common import write_json


class MetadataExporter:
    def __init__(self, report_dir: Path):
        self.report_dir = report_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, payload: Dict[str, Any]) -> None:
        write_json(self.report_dir / name, payload)

    def write_markdown(self, name: str, content: str) -> None:
        (self.report_dir / name).write_text(content, encoding="utf-8")
