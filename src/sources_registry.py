from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import yaml

@dataclass
class Source:
    id: str
    kind: str
    dataset: str | None = None
    url: str | None = None
    out_dir: str | None = None
    notes: str | None = None

def load_sources(path: str | Path = "data_sources.yaml") -> List[Source]:
    path = Path(path)
    if not path.exists():
        # Fallback to looking in root if we are in src/ or similar
        root_path = Path(".").resolve().parent / path.name
        if root_path.exists():
            path = root_path
        elif (Path("..") / path.name).exists():
             path = Path("..") / path.name

    obj: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    sources = []
    for s in obj.get("sources", []):
        sources.append(Source(**s))
    return sources
