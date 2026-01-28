from __future__ import annotations

import json
import os
from pathlib import Path


def ensure_kaggle_auth() -> bool:
    kaggle_json = os.getenv("KAGGLE_JSON", "").strip()
    username = os.getenv("KAGGLE_USERNAME", "").strip()
    key = os.getenv("KAGGLE_KEY", "").strip()

    if not kaggle_json and (not username or not key):
        print("Kaggle secrets missing, skipping")
        return False

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_path = kaggle_dir / "kaggle.json"

    if kaggle_json:
        try:
            payload = json.loads(kaggle_json)
        except json.JSONDecodeError:
            payload = {"username": username, "key": key}
    else:
        payload = {"username": username, "key": key}

    if not payload.get("username") or not payload.get("key"):
        print("Kaggle secrets missing, skipping")
        return False

    kaggle_path.write_text(json.dumps(payload))
    os.chmod(kaggle_path, 0o600)
    return True
