from pathlib import Path
import pandas as pd
import numpy as np

RAW = Path("data/raw")
ART = Path("artifacts")

def _read_csv(name: str) -> pd.DataFrame:
    p = RAW / name
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run player-query-bundle first.")
    return pd.read_csv(p)

def _pick(df, options):
    cols = {c.lower(): c for c in df.columns}
    for o in options:
        if o in cols: return cols[o]
    for c in df.columns:
        cl = c.lower()
        for o in options:
            if o in cl: return c
    return None

def build_features():
    ART.mkdir(parents=True, exist_ok=True)

    pb = _read_csv("nba_player_box.csv")
    tb = _read_csv("nba_team_box.csv")

    # Standardize minimal columns
    player_c = _pick(pb, ["player", "player_name", "athlete", "athlete_display_name", "name"])
    date_c   = _pick(pb, ["game_date", "date", "start_date", "game_datetime"])
    team_c   = _pick(pb, ["team", "team_abbreviation", "team_abbr", "team_id"])
    opp_c    = _pick(pb, ["opponent", "opp", "opponent_abbreviation", "opp_abbr"])
    min_c    = _pick(pb, ["min", "minutes", "mp"])
    pts_c    = _pick(pb, ["pts", "points"])
    reb_c    = _pick(pb, ["reb", "rebounds", "trb", "rebs"])
    ast_c    = _pick(pb, ["ast", "assists"])

    needed = {"player":player_c,"game_date":date_c,"team":team_c,"opp":opp_c,"min":min_c,"pts":pts_c,"reb":reb_c,"ast":ast_c}
    missing = [k for k,v in needed.items() if v is None]
    if missing:
        raise ValueError(f"Player box missing columns: {missing}. Have: {list(pb.columns)[:50]}")

    pb = pb.rename(columns={
        player_c:"player", date_c:"game_date", team_c:"team", opp_c:"opp",
        min_c:"min", pts_c:"pts", reb_c:"reb", ast_c:"ast"
    }).copy()

    pb["game_date"] = pd.to_datetime(pb["game_date"], errors="coerce")
    pb = pb.dropna(subset=["game_date"])
    for c in ["min","pts","reb","ast"]:
        pb[c] = pd.to_numeric(pb[c], errors="coerce").fillna(0.0)

    # Rolling player features (no leakage)
    pb = pb.sort_values(["player","game_date"])
    g = pb.groupby("player", group_keys=False)
    for s in ["min","pts","reb","ast"]:
        pb[f"{s}_r5"]   = g[s].shift(1).rolling(5, min_periods=1).mean()
        pb[f"{s}_r10"]  = g[s].shift(1).rolling(10, min_periods=1).mean()
        pb[f"{s}_sd10"] = g[s].shift(1).rolling(10, min_periods=2).std()
    pb["gp_last14"] = g["game_date"].shift(1).rolling(14, min_periods=1).count()
    pb = pb.fillna(0.0)

    # Team context from team box (light auto-detect)
    tb_date = _pick(tb, ["game_date", "date", "start_date", "game_datetime"])
    tb_team = _pick(tb, ["team", "team_abbreviation", "team_abbr", "team_id"])
    ortg    = _pick(tb, ["ortg", "off_rating", "offensive_rating"])
    drtg    = _pick(tb, ["drtg", "def_rating", "defensive_rating"])
    pace    = _pick(tb, ["pace"])

    if tb_date and tb_team:
        t = tb.rename(columns={tb_date:"game_date", tb_team:"team"}).copy()
        t["game_date"] = pd.to_datetime(t["game_date"], errors="coerce")
        t = t.dropna(subset=["game_date"])
        keep = ["game_date","team"]
        ren = {}
        if ortg: ren[ortg] = "team_ortg"
        if drtg: ren[drtg] = "team_drtg"
        if pace: ren[pace] = "team_pace"
        t = t.rename(columns=ren)
        keep += list(ren.values())
        t = t[keep].drop_duplicates(subset=["game_date","team"])
        pb = pb.merge(t, on=["game_date","team"], how="left")
    else:
        pb["team_ortg"] = 0.0
        pb["team_drtg"] = 0.0
        pb["team_pace"] = 0.0

    pb[["team_ortg","team_drtg","team_pace"]] = pb[["team_ortg","team_drtg","team_pace"]].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    out = ART / "features_v1.parquet"
    pb.to_parquet(out, index=False)
    print("Saved features:", out, "rows", len(pb), "cols", pb.shape[1])

if __name__ == "__main__":
    build_features()
