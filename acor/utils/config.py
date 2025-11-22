import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if overrides:
        for key, value in overrides.items():
            assign_dotted_key(config, key, value)
    return config


def assign_dotted_key(base: Dict[str, Any], dotted_key: str, value: Any) -> None:
    segments = dotted_key.split(".")
    current = base
    for seg in segments[:-1]:
        if seg not in current or not isinstance(current[seg], dict):
            current[seg] = {}
        current = current[seg]
    current[segments[-1]] = value


def parse_unknown_args(unknown: list[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for arg in unknown:
        if not arg.startswith("--"):
            continue
        key, _, raw_value = arg[2:].partition("=")
        overrides[key] = _infer_value(raw_value)
    return overrides


def _infer_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def build_argparser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional output directory.")
    parser.add_argument("--run_name", type=str, default=None, help="Optional run identifier.")
    return parser
