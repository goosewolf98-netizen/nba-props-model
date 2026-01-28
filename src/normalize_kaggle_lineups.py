from __future__ import annotations

from pathlib import Path

import pandas as pd

from schema_normalize import norm_all


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


def _lineup_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "player1",
        "player2",
        "player3",
        "player4",
        "player5",
        "player_1",
        "player_2",
        "player_3",
        "player_4",
        "player_5",
        "player_on_1",
        "player_on_2",
        "player_on_3",
        "player_on_4",
        "player_on_5",
    ]
    return [c for c in candidates if c in df.columns]


def main() -> None:
    base_dir = Path("data/kaggle/xocelyk/nba-pbp")
    if not base_dir.exists():
        print("Kaggle lineups not found, skipping")
        return

    frames = _load_kaggle_files(base_dir)
    if not frames:
        print("No Kaggle lineups files found, skipping")
        return

    output_frames: list[pd.DataFrame] = []
    for frame in frames:
        if frame.empty:
            continue
        frame = norm_all(frame)
        if "game_id" not in frame.columns:
            for alt in ["GAME_ID", "gameId", "game_id"]:
                if alt in frame.columns:
                    frame = frame.rename(columns={alt: "game_id"})
                    break
        lineup_cols = _lineup_columns(frame)
        if "lineup_id" not in frame.columns and lineup_cols:
            frame["lineup_id"] = frame[lineup_cols].astype(str).agg("|".join, axis=1)
        possessions_col = None
        for col in ["possessions", "possessions_total", "num_possessions", "possessions_off", "possessions_def"]:
            if col in frame.columns:
                possessions_col = col
                break
        minutes_col = None
        for col in ["minutes", "min", "seconds", "secs", "duration_seconds"]:
            if col in frame.columns:
                minutes_col = col
                break
        if possessions_col is None and minutes_col is None:
            continue

        out = frame.copy()
        out["possessions_proxy"] = pd.to_numeric(out.get(possessions_col), errors="coerce") if possessions_col else 0
        if minutes_col:
            mins = pd.to_numeric(out.get(minutes_col), errors="coerce")
            if minutes_col in {"seconds", "secs", "duration_seconds"}:
                mins = mins / 60.0
            out["minutes_proxy"] = mins
        else:
            out["minutes_proxy"] = 0

        keep_cols = ["game_date", "game_id", "team_abbr", "opp_abbr", "lineup_id"]
        if lineup_cols:
            keep_cols += lineup_cols
        keep_cols += ["possessions_proxy", "minutes_proxy"]
        out = out.reindex(columns=[c for c in keep_cols if c in out.columns])
        output_frames.append(out)

    if not output_frames:
        print("No usable Kaggle lineups rows, skipping")
        return

    lineups = pd.concat(output_frames, ignore_index=True)
    lineups = lineups.dropna(subset=["game_date", "team_abbr"], how="any")
    if lineups.empty:
        print("No Kaggle lineups rows after cleanup, skipping")
        return

    group_cols = ["game_date", "game_id", "team_abbr", "opp_abbr", "lineup_id"]
    lineup_cols = _lineup_columns(lineups)
    if lineup_cols:
        group_cols += lineup_cols
    group_cols = [c for c in group_cols if c in lineups.columns]
    lineups = lineups.groupby(group_cols, as_index=False).agg(
        possessions_proxy=("possessions_proxy", "sum"),
        minutes_proxy=("minutes_proxy", "sum"),
    )

    output_path = Path("data/raw/kaggle_pbp_lineups.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lineups.to_parquet(output_path, index=False)
    print("Saved", output_path)


if __name__ == "__main__":
    main()
