from __future__ import annotations

from pathlib import Path
import shutil

from kaggle.api.kaggle_api_extended import KaggleApi

from kaggle_auth import ensure_kaggle_auth

DATASETS = [
    "xocelyk/nba-pbp",
    "mexwell/nba-shots",
    "wyattowalsh/basketball",
    "loganlauton/nba-injury-stats-1951-2023",
    "jacquesoberweis/2016-2025-nba-injury-data",
    "cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024",
    "christophertreasure/nba-odds-data",
]


def _slug_to_dir(slug: str) -> Path:
    return Path(*slug.split("/"))


def main():
    if not ensure_kaggle_auth():
        return

    base_dir = Path("data/kaggle")
    base_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        print(f"Kaggle auth failed, skipping: {exc}")
        return

    for dataset in DATASETS:
        out_dir = base_dir / _slug_to_dir(dataset)
        out_dir.mkdir(parents=True, exist_ok=True)
        if any(out_dir.iterdir()):
            print(f"Kaggle dataset already cached: {dataset}")
            continue
        tmp_dir = out_dir / "_tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Kaggle dataset: {dataset}")
        try:
            api.dataset_download_files(dataset, path=str(tmp_dir), unzip=True, quiet=False)
        except Exception as exc:
            print(f"Kaggle download failed for {dataset}: {exc}")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            continue
        for item in tmp_dir.iterdir():
            shutil.move(str(item), out_dir / item.name)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("Saved Kaggle dataset to", out_dir)


if __name__ == "__main__":
    main()
