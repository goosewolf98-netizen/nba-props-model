from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
ART = Path("artifacts")

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the bundle download step first.")
    return pd.read_csv(path)

def _pick(df: pd.DataFrame, options):
    cols = {c.lower(): c for c in df.columns}
    for o in options:
        if o in cols:
            return cols[o]
    for c in df.columns:
        cl = c.lower()
        for o in options:
            if o in cl:
                return c
    return None

def build_features():
    ART.mkdir(parents=True, exist_ok=True)

    pb = _read_csv(RAW / "nba_player_box.csv")
    tb = _read_csv(RAW / "nba_team_box.csv")

    # --- standardize player box ---
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
        raise ValueError(f"nba_player_box.csv missing columns: {missing}. Have: {list(pb.columns)[:50]}")

    pb = pb.rename(columns={
        player_c:"player", date_c:"game_date", team_c:"team", opp_c:"opp",
        min_c:"min", pts_c:"pts", reb_c:"reb", ast_c:"ast"
    }).copy()

    pb["game_date"] = pd.to_datetime(pb["game_date"], errors="coerce")
    pb = pb.dropna(subset=["game_date"])

    for c in ["min","pts","reb","ast"]:
        pb[c] = pd.to_numeric(pb[c], errors="coerce").fillna(0.0)

    # --- rolling player features (no leakage) ---
    pb = pb.sort_values(["player","game_date"])
    g = pb.groupby("player", group_keys=False)

    for s in ["min","pts","reb","ast"]:
        pb[f"{s}_r5"]   = g[s].shift(1).rolling(5, min_periods=1).mean()
        pb[f"{s}_r10"]  = g[s].shift(1).rolling(10, min_periods=1).mean()
        pb[f"{s}_sd10"] = g[s].shift(1).rolling(10, min_periods=2).std()

    pb["gp_last14"] = g["game_date"].shift(1).rolling(14, min_periods=1).count()
    pb = pb.fillna(0.0)

    # --- team context (SAFE: default to 0s if not found) ---
    tb_date = _pick(tb, ["game_date", "date", "start_date", "game_datetime"])
    tb_team = _pick(tb, ["team", "team_abbreviation", "team_abbr", "team_id"])
    ortg    = _pick(tb, ["ortg", "off_rating", "offensive_rating"])
    drtg    = _pick(tb, ["drtg", "def_rating", "defensive_rating"])
    pace    = _pick(tb, ["pace"])

    pb["team_ortg"] = 0.0
    pb["team_drtg"] = 0.0
    pb["team_pace"] = 0.0

    if tb_date and tb_team:
        t = tb.rename(columns={tb_date:"game_date", tb_team:"team"}).copy()
        t["game_date"] = pd.to_datetime(t["game_date"], errors="coerce")
        t = t.dropna(subset=["game_date"])

        ren = {}
        if ortg: ren[ortg] = "team_ortg"
        if drtg: ren[drtg] = "team_drtg"
        if pace: ren[pace] = "team_pace"

        if ren:
            t = t.rename(columns=ren)
            t = t[["game_date","team"] + list(ren.values())].drop_duplicates(subset=["game_date","team"])
            pb = pb.merge(t, on=["game_date","team"], how="left")

    for col in ["team_ortg","team_drtg","team_pace"]:
        pb[col] = pd.to_numeric(pb[col], errors="coerce").fillna(0.0)

    # ✅ IMPORTANT: keep only the columns we need (prevents Parquet type errors)
    engineered = [c for c in pb.columns if c.endswith(("_r5","_r10","_sd10"))] + ["gp_last14","team_ortg","team_drtg","team_pace"]
    keep_cols = ["player","game_date","team","opp","min","pts","reb","ast"] + engineered
    pb = pb[keep_cols].copy()

    out = ART / "features_v1.parquet"
    pb.to_parquet(out, index=False)
    print("Saved features:", out, "rows", len(pb), "cols", pb.shape[1])

if __name__ == "__main__":
    build_features()
