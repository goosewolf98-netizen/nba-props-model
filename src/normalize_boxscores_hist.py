from __future__ import annotations

from pathlib import Path
import pandas as pd

from schema_normalize import norm_all, norm_minutes


OUT_PATH = Path("data/raw/player_box_hist.parquet")
KAGGLE_DIR = Path("data/kaggle/szymonjwiak__nba-scoring-boxscores-1997-2023")


def _find_source_files() -> list[Path]:
    if not KAGGLE_DIR.exists():
        return []
    candidates = list(KAGGLE_DIR.rglob("*.parquet")) + list(KAGGLE_DIR.rglob("*.csv"))
    player_files = [p for p in candidates if "player" in p.name.lower()]
    return player_files or candidates


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
            "player",
            "team_abbr",
            "opp_abbr",
            "min",
            "pts",
            "reb",
            "ast",
            "usage_proxy",
        ]).to_parquet(OUT_PATH, index=False)
        print("Missing historical boxscores; wrote empty player_box_hist.parquet")
        return

    df = norm_all(df)
    df = norm_minutes(df)
    cols_lower = {c.lower(): c for c in df.columns}
    game_id_col = cols_lower.get("game_id") or cols_lower.get("gameid") or cols_lower.get("gamecode")
    player_col = cols_lower.get("player") or cols_lower.get("player_name") or cols_lower.get("name")
    pts_col = cols_lower.get("pts") or cols_lower.get("points")
    reb_col = cols_lower.get("reb") or cols_lower.get("rebounds")
    ast_col = cols_lower.get("ast") or cols_lower.get("assists")
    min_col = cols_lower.get("min") or cols_lower.get("minutes")

    out = pd.DataFrame()
    out["game_date"] = df.get("game_date", "")
    out["game_id"] = df[game_id_col] if game_id_col else ""
    out["player"] = df[player_col] if player_col else ""
    out["team_abbr"] = df.get("team_abbr", "")
    out["opp_abbr"] = df.get("opp_abbr", "")
    out["min"] = pd.to_numeric(df[min_col], errors="coerce").fillna(0.0) if min_col else 0.0
    out["pts"] = pd.to_numeric(df[pts_col], errors="coerce").fillna(0.0) if pts_col else 0.0
    out["reb"] = pd.to_numeric(df[reb_col], errors="coerce").fillna(0.0) if reb_col else 0.0
    out["ast"] = pd.to_numeric(df[ast_col], errors="coerce").fillna(0.0) if ast_col else 0.0
    out["usage_proxy"] = pd.to_numeric(df.get("usage", 0), errors="coerce").fillna(0.0) if "usage" in df.columns else 0.0

    out = out.dropna(subset=["game_date"]).copy()
    out.to_parquet(OUT_PATH, index=False)
    print("Saved historical player boxscores to", OUT_PATH)


if __name__ == "__main__":
    main()
