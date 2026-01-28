from __future__ import annotations

from pathlib import Path

import pandas as pd

from schema_normalize import norm_all


SHOT_PROFILE_VERSION = "v1"


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


def _normalize_player(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["player", "player_name", "player_name_x", "PLAYER_NAME"]:
        if col in df.columns:
            df = df.rename(columns={col: "player"})
            break
    return df


def _normalize_team(df: pd.DataFrame) -> pd.DataFrame:
    if "team_abbr" not in df.columns:
        for col in ["team", "team_abbreviation", "TEAM_ABBR", "team_name"]:
            if col in df.columns:
                df = df.rename(columns={col: "team_abbr"})
                break
    return df


def _infer_zone(df: pd.DataFrame) -> pd.Series:
    if "shot_zone_basic" in df.columns:
        zone = df["shot_zone_basic"].astype(str).str.lower()
        return zone
    if "shot_type" in df.columns:
        zone = df["shot_type"].astype(str).str.lower()
        return zone
    return pd.Series([""] * len(df))


def _infer_distance(df: pd.DataFrame) -> pd.Series:
    for col in ["shot_distance", "distance", "shot_dist"]:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([pd.NA] * len(df))


def main() -> None:
    base_dir = Path("data/kaggle/mexwell/nba-shots")
    if not base_dir.exists():
        print("Kaggle shots not found, skipping")
        return

    frames = _load_kaggle_files(base_dir)
    if not frames:
        print("No Kaggle shots files found, skipping")
        return

    output_frames: list[pd.DataFrame] = []
    for frame in frames:
        if frame.empty:
            continue
        frame = norm_all(frame)
        frame = _normalize_player(frame)
        frame = _normalize_team(frame)
        if "player" not in frame.columns or "team_abbr" not in frame.columns:
            continue

        made_col = None
        for col in ["shot_made_flag", "shot_made", "made", "fg_made"]:
            if col in frame.columns:
                made_col = col
                break
        if made_col is None:
            continue

        zone = _infer_zone(frame)
        distance = _infer_distance(frame)

        def classify_shot(row_zone: str, row_distance) -> str:
            text = row_zone
            if "3" in text or "three" in text:
                return "three"
            if "restricted" in text or "rim" in text:
                return "rim"
            if "mid" in text:
                return "mid"
            if pd.notna(row_distance):
                if row_distance <= 4:
                    return "rim"
                if row_distance <= 22:
                    return "mid"
                return "three"
            return "mid"

        shot_zone = [
            classify_shot(z, d)
            for z, d in zip(zone.fillna(""), distance)
        ]

        out = frame[["game_date", "player", "team_abbr"]].copy()
        out["shot_zone"] = shot_zone
        out["made"] = pd.to_numeric(frame[made_col], errors="coerce").fillna(0).astype(int)
        output_frames.append(out)

    if not output_frames:
        print("No usable Kaggle shots rows, skipping")
        return

    shots = pd.concat(output_frames, ignore_index=True)
    shots = shots.dropna(subset=["game_date", "player", "team_abbr"], how="any")
    if shots.empty:
        print("No Kaggle shots rows after cleanup, skipping")
        return

    agg = shots.groupby(["game_date", "player", "team_abbr", "shot_zone"], as_index=False).agg(
        attempts=("shot_zone", "size"),
        makes=("made", "sum"),
    )

    pivot_att = agg.pivot_table(
        index=["game_date", "player", "team_abbr"],
        columns="shot_zone",
        values="attempts",
        fill_value=0,
    ).reset_index()
    pivot_mk = agg.pivot_table(
        index=["game_date", "player", "team_abbr"],
        columns="shot_zone",
        values="makes",
        fill_value=0,
    ).reset_index()

    merged = pivot_att.merge(pivot_mk, on=["game_date", "player", "team_abbr"], suffixes=("_att", "_fg"))

    for zone in ["rim", "mid", "three"]:
        if f"{zone}_att" not in merged.columns:
            merged[f"{zone}_att"] = 0
        if f"{zone}_fg" not in merged.columns:
            merged[f"{zone}_fg"] = 0

    merged["shot_profile_version"] = SHOT_PROFILE_VERSION

    output_path = Path("data/raw/kaggle_shots_agg.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    print("Saved", output_path)


if __name__ == "__main__":
    main()
