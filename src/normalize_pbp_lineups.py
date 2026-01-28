from __future__ import annotations

from pathlib import Path
import pandas as pd

from schema_normalize import norm_all


OUT_PATH = Path("data/raw/pbp_lineups.parquet")
KAGGLE_DIR = Path("data/kaggle/xocelyk__nba-pbp")


def _find_source_files() -> list[Path]:
    if not KAGGLE_DIR.exists():
        return []
    candidates = list(KAGGLE_DIR.rglob("*.parquet")) + list(KAGGLE_DIR.rglob("*.csv"))
    lineup_files = [p for p in candidates if "lineup" in p.name.lower()]
    return lineup_files or candidates


def _read_any(paths: list[Path]) -> pd.DataFrame:
    for path in paths:
        try:
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
            return pd.read_csv(path)
        except Exception:
            continue
    return pd.DataFrame()


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sources = _find_source_files()
    df = _read_any(sources)
    if df.empty:
        pd.DataFrame(columns=[
            "game_date",
            "game_id",
            "team_abbr",
            "opp_abbr",
            "lineup_id",
            "player_on_1",
            "player_on_2",
            "player_on_3",
            "player_on_4",
            "player_on_5",
            "player_off_1",
            "player_off_2",
            "player_off_3",
            "player_off_4",
            "player_off_5",
            "possessions_proxy",
            "minutes_proxy",
        ]).to_parquet(OUT_PATH, index=False)
        print("Missing lineup data; wrote empty pbp_lineups.parquet")
        return

    df = norm_all(df)
    cols_lower = {c.lower(): c for c in df.columns}
    game_id_col = cols_lower.get("game_id") or cols_lower.get("gameid") or cols_lower.get("gamecode")
    lineup_id_col = cols_lower.get("lineup_id") or cols_lower.get("lineupid")

    player_on_cols = [c for c in df.columns if "player_on" in c.lower()]
    player_off_cols = [c for c in df.columns if "player_off" in c.lower()]
    if not player_on_cols:
        player_on_cols = [c for c in df.columns if "player" in c.lower()][:5]

    out = pd.DataFrame()
    out["game_date"] = df.get("game_date", "")
    if game_id_col:
        out["game_id"] = df[game_id_col]
    else:
        out["game_id"] = ""
    out["team_abbr"] = df.get("team_abbr", "")
    out["opp_abbr"] = df.get("opp_abbr", "")
    out["lineup_id"] = df[lineup_id_col] if lineup_id_col else ""

    for i in range(5):
        col = player_on_cols[i] if i < len(player_on_cols) else None
        out[f"player_on_{i+1}"] = df[col] if col else ""
    for i in range(5):
        col = player_off_cols[i] if i < len(player_off_cols) else None
        out[f"player_off_{i+1}"] = df[col] if col else ""

    poss_col = cols_lower.get("possessions") or cols_lower.get("poss") or cols_lower.get("possession")
    mins_col = cols_lower.get("minutes") or cols_lower.get("min")
    out["possessions_proxy"] = pd.to_numeric(df[poss_col], errors="coerce").fillna(0.0) if poss_col else 0.0
    out["minutes_proxy"] = pd.to_numeric(df[mins_col], errors="coerce").fillna(0.0) if mins_col else 0.0

    out = out.dropna(subset=["game_date"]).copy()
    out.to_parquet(OUT_PATH, index=False)
    print("Saved pbp lineups to", OUT_PATH)


if __name__ == "__main__":
    main()
