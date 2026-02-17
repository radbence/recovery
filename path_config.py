from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).with_name("file_grouping_config.json")


def load_config_section(config_path: Path, section_name: str) -> dict[str, Any]:
    if not config_path.exists():
        return {}

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"WARNING: Could not parse config file {config_path}: {exc}")
        return {}

    if not isinstance(data, dict):
        return {}

    section = data.get(section_name, {})
    if isinstance(section, dict):
        return section

    return {}
