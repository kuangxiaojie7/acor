from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .distributed import is_main_process


class JSONLogger:
    """
    Minimal JSONL logger mirroring output to stdout.
    """

    def __init__(self, output_dir: Optional[str] = None, filename: str = "metrics.jsonl") -> None:
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._path = self.output_dir / filename
            self._handle = self._path.open("a", encoding="utf-8")
        else:
            self._path = None
            self._handle = None

    def log(self, payload: Dict[str, Any]) -> None:
        if not is_main_process():
            return
        enriched = {"timestamp": dt.datetime.utcnow().isoformat() + "Z", **payload}
        line = json.dumps(enriched, ensure_ascii=False)
        print(line, file=sys.stdout, flush=True)
        if self._handle:
            self._handle.write(line + os.linesep)
            self._handle.flush()

    def close(self) -> None:
        if self._handle:
            self._handle.close()


def format_metrics(step: int, metrics: Dict[str, Any]) -> str:
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return f"[step={step}] " + ", ".join(parts)
