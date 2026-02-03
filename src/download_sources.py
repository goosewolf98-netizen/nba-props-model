from __future__ import annotations

import os
import json
import shutil
import subprocess
from pathlib import Path
from typing import List

import requests

from sources_registry import load_sources, Source

ROOT = Path(".").resolve()

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def ensure_kaggle_auth() -> bool:
    """
    Ensures ~/.kaggle/kaggle.json exists, using env vars if available.
    """
    kaggle_json = os.getenv("KAGGLE_JSON", "").strip()
    username = os.getenv("KAGGLE_USERNAME", "").strip()
    key = os.getenv("KAGGLE_KEY", "").strip()

    # If ~/.kaggle/kaggle.json already exists, we might trust it,
    # but CI environments usually need us to write it from env vars.
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_path = kaggle_dir / "kaggle.json"

    # If no env vars and file exists, assume it's good (local dev)
    if not kaggle_json and (not username or not key):
        if kaggle_path.exists():
            return True
        print("Kaggle secrets missing (KAGGLE_JSON or KAGGLE_USERNAME+KAGGLE_KEY) and no ~/.kaggle/kaggle.json found.")
        return False

    ensure_dir(kaggle_dir)

    if kaggle_json:
        try:
            payload = json.loads(kaggle_json)
        except json.JSONDecodeError:
            # Fallback if it's not valid JSON, though it should be
            payload = {"username": username, "key": key}
    else:
        payload = {"username": username, "key": key}

    if not payload.get("username") or not payload.get("key"):
        print("Kaggle secrets incomplete.")
        return False

    kaggle_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        os.chmod(kaggle_path, 0o600)
    except Exception:
        pass
    return True

def kaggle_download(dataset: str, out_dir: Path, force: bool = False) -> None:
    ensure_dir(out_dir)

    if not force and out_dir.exists() and any(out_dir.iterdir()):
        print(f"Directory not empty, skipping download for: {dataset}")
        return

    print(f"Downloading Kaggle dataset: {dataset} -> {out_dir}")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError("kaggle package not installed. Run 'pip install kaggle'")

    api = KaggleApi()
    api.authenticate()

    # Download to a temp dir first to avoid partial state
    tmp_dir = out_dir / "_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    ensure_dir(tmp_dir)

    try:
        api.dataset_download_files(dataset, path=str(tmp_dir), unzip=True, quiet=False)
    except Exception as exc:
        print(f"Kaggle download failed for {dataset}: {exc}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise exc

    # Move files from tmp to out_dir
    for item in tmp_dir.iterdir():
        # Avoid moving the tmp dir itself if it iterates over it (it shouldn't)
        if item.name == "_tmp": continue
        dest = out_dir / item.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(item), str(dest))

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"Saved {dataset}")

def github_zip_download(url: str, out_dir: Path, force: bool = False) -> None:
    ensure_dir(out_dir)

    if not force and any(out_dir.iterdir()):
        print(f"Directory not empty, skipping GitHub download: {url}")
        return

    print(f"Downloading GitHub zip: {url} -> {out_dir}")
    zip_path = out_dir / "repo.zip"

    try:
        r = requests.get(url, timeout=60, stream=True)
        r.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        if zip_path.exists():
            zip_path.unlink()
        return

    # unzip into out_dir/unzipped or just root?
    # The prompt suggested: unzip into out_dir/unzipped
    unzipped = out_dir / "unzipped"
    if unzipped.exists():
        shutil.rmtree(unzipped)
    ensure_dir(unzipped)

    try:
        shutil.unpack_archive(str(zip_path), str(unzipped))
    except Exception as e:
        print(f"Failed to unpack {zip_path}: {e}")
        return
    finally:
        zip_path.unlink(missing_ok=True)

    print(f"Extracted to {unzipped}")


def run_selected(sources: List[Source], only: List[str] | None = None, force: bool = False) -> None:
    for s in sources:
        if only and s.id not in only:
            continue

        out_dir = ROOT / (s.out_dir or f"data/external/{s.id}")
        print(f"\n=== {s.id} [{s.kind}] ===")

        try:
            if s.kind == "kaggle_dataset":
                ok = ensure_kaggle_auth()
                if not ok:
                    print("Skipping (Auth failed)")
                    continue
                kaggle_download(s.dataset, out_dir, force=force)

            elif s.kind == "github_zip":
                github_zip_download(s.url, out_dir, force=force)

            elif s.kind == "snapshot_placeholder":
                ensure_dir(out_dir)
                readme = out_dir / "README.txt"
                if not readme.exists():
                    readme.write_text(
                        "Placeholder: we will add daily prop-lines snapshot collector here.\n",
                        encoding="utf-8"
                    )
                print(f"Checked placeholder at {out_dir}")

            else:
                print(f"Unknown source kind: {s.kind}")
        except Exception as e:
            print(f"Error processing {s.id}: {e}")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="Optional list of source IDs to run")
    ap.add_argument("--force", action="store_true", help="Force redownload (not fully implemented yet, relies on dir empty check)")
    args = ap.parse_args()

    sources = load_sources()
    run_selected(sources, only=args.only, force=args.force)

if __name__ == "__main__":
    main()
