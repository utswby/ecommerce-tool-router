from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_PATH = PROJECT_ROOT / "data" / "tools.json"


def load_tools() -> list[dict[str, Any]]:
    with TOOLS_PATH.open(encoding="utf-8") as source:
        tools = json.load(source)
    names = [tool.get("name") for tool in tools]
    if not tools or len(names) != len(set(names)):
        raise ValueError("Tool catalog must contain unique tool names")
    return tools


def tool_document(tool: dict[str, Any]) -> str:
    fields = [tool["name"], tool["description"], *tool.get("keywords", [])]
    return " ".join(part.strip() for part in fields if isinstance(part, str) and part.strip())
