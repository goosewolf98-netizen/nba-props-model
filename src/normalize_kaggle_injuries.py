from __future__ import annotations

from pathlib import Path

import pandas as pd


def _load_kaggle_files(base_dir: Path) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for path in base_dir.rglob("*"):
        if path.suffix.lower() == ".csv":
            try:
                frames.append(pd.read_csv(path))
            except Exception:
                continue
        elif path.suffix.lower() == ".parquet":
            try:
                frames.append(pd.read_parquet(path))
            except Exception:
                continue
    return frames


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["player", "Player", "PLAYER", "Name", "player_name"]:
        if col in df.columns:
            df = df.rename(columns={col: "player"})
            break
    for col in ["team_abbr", "team", "Team", "TEAM", "team_abbreviation"]:
        if col in df.columns:
            df = df.rename(columns={col: "team_abbr"})
            break
    for col in ["injury_type", "Injury", "injury", "Injury Type", "injury_desc"]:
        if col in df.columns:
            df = df.rename(columns={col: "injury_type"})
            break
    for col in ["start_date", "Injury Date", "injury_date", "Date", "from", "start"]:
        if col in df.columns:
            df = df.rename(columns={col: "start_date"})
            break
    for col in ["end_date", "Return Date", "return_date", "to", "end"]:
        if col in df.columns:
            df = df.rename(columns={col: "end_date"})
            break
    for col in ["games_missed", "Games Missed", "games_missed_count"]:
        if col in df.columns:
            df = df.rename(columns={col: "games_missed"})
            break
    return df


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    if "start_date" in df.columns:
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce").dt.date.astype(str)
    if "end_date" in df.columns:
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce").dt.date.astype(str)
    return df


def main() -> None:
    base_dirs = [
        Path("data/kaggle/loganlauton/nba-injury-stats-1951-2023"),
        Path("data/kaggle/jacquesoberweis/2016-2025-nba-injury-data"),
    ]

    frames: list[pd.DataFrame] = []
    for base_dir in base_dirs:
        if not base_dir.exists():
            continue
        frames.extend(_load_kaggle_files(base_dir))

    if not frames:
        print("Kaggle injury history not found, skipping")
        return

    output_frames: list[pd.DataFrame] = []
    for frame in frames:
        if frame.empty:
            continue
        frame = _normalize_columns(frame)
        frame = _normalize_dates(frame)
        if "player" not in frame.columns or "start_date" not in frame.columns:
            continue
        out = frame[[c for c in ["player", "start_date", "end_date", "team_abbr", "injury_type", "games_missed"] if c in frame.columns]].copy()
        output_frames.append(out)

    if not output_frames:
        print("No usable Kaggle injury rows, skipping")
        return

    injuries = pd.concat(output_frames, ignore_index=True)
    injuries = injuries.dropna(subset=["player", "start_date"], how="any")
    if "games_missed" in injuries.columns:
        injuries["games_missed"] = pd.to_numeric(injuries["games_missed"], errors="coerce")

    output_path = Path("data/raw/kaggle_injuries_history.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    injuries.to_parquet(output_path, index=False)
    print("Saved", output_path)


if __name__ == "__main__":
    main()
