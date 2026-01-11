from pathlib import Path
import pandas as pd
import numpy as np

RAW = Path("data/raw")
ART = Path("artifacts")

def _read_csv(name: str) -> pd.DataFrame:
    p = RAW / name
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}.")
    return pd.read_csv(p)

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

def _team_rest_table(team_box: pd.DataFrame):
    tb = team_box.copy()

    tb_date = _pick(tb, ["game_date", "date", "start_date", "game_datetime"])
    tb_team = _pick(tb, ["team", "team_abbreviation", "team_abbr", "team_id", "team_slug"])

    if tb_date is None or tb_team is None:
        return None

    tb = tb.rename(columns={tb_date: "game_date", tb_team: "team"}).copy()
    tb["game_date"] = pd.to_datetime(tb["game_date"], errors="coerce")
    tb = tb.dropna(subset=["game_date"])
    tb["team"] = tb["team"].astype(str)

    tb = tb.sort_values(["team", "game_date"])
    tb["rest_days"] = tb.groupby("team")["game_date"].diff().dt.days
    tb["rest_days"] = tb["rest_days"].fillna(7).clip(lower=0, upper=14)

    tb["b2b"] = (tb["rest_days"] <= 1).astype(int)

    # games played in prior 7 days (excluding current)
    def games_last_7d(g):
        g = g.sort_values("game_date").copy()
        g = g.set_index("game_date")
        # rolling count in last 7 days INCLUDING current, then subtract 1
        c = g["b2b"].rolling("7D").count().fillna(0) - 1
        g["games_last_7d"] = c.clip(lower=0).values
        return g.reset_index()

    tb = tb.groupby("team", group_keys=False).apply(games_last_7d)

    return tb[["game_date", "team", "rest_days", "b2b", "games_last_7d"]].drop_duplicates(["game_date", "team"])

def build_features():
    ART.mkdir(parents=True, exist_ok=True)

    pb = _read_csv("nba_player_box.csv")
    tb = _read_csv("nba_team_box.csv")

    # --- standardize player box ---
    player_c = _pick(pb, ["player", "player_name", "athlete", "athlete_display_name", "name"])
    date_c   = _pick(pb, ["game_date", "date", "start_date", "game_datetime"])
    team_c   = _pick(pb, ["team", "team_abbreviation", "team_abbr", "team_id", "team_slug"])
    opp_c    = _pick(pb, ["opponent", "opp", "opponent_abbreviation", "opp_abbr", "opponent_team"])
    min_c    = _pick(pb, ["min", "minutes", "mp"])
    pts_c    = _pick(pb, ["pts", "points"])
    reb_c    = _pick(pb, ["reb", "rebounds", "trb", "rebs"])
    ast_c    = _pick(pb, ["ast", "assists"])

    needed = {"player":player_c,"game_date":date_c,"team":team_c,"opp":opp_c,"min":min_c,"pts":pts_c,"reb":reb_c,"ast":ast_c}
    missing = [k for k,v in needed.items() if v is None]
    if missing:
        raise ValueError(f"nba_player_box.csv missing columns: {missing}. Have: {list(pb.columns)[:60]}")

    pb = pb.rename(columns={
        player_c:"player", date_c:"game_date", team_c:"team", opp_c:"opp",
        min_c:"min", pts_c:"pts", reb_c:"reb", ast_c:"ast"
    }).copy()

    pb["game_date"] = pd.to_datetime(pb["game_date"], errors="coerce")
    pb = pb.dropna(subset=["game_date"])

    pb["player"] = pb["player"].astype(str)
    pb["team"] = pb["team"].astype(str)
    pb["opp"] = pb["opp"].astype(str)

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

    # --- team metrics (if detectable) ---
    tb_date = _pick(tb, ["game_date", "date", "start_date", "game_datetime"])
    tb_team = _pick(tb, ["team", "team_abbreviation", "team_abbr", "team_id", "team_slug"])
    ortg    = _pick(tb, ["ortg", "off_rating", "offensive_rating"])
    drtg    = _pick(tb, ["drtg", "def_rating", "defensive_rating"])
    pace    = _pick(tb, ["pace"])

    # defaults (always exist)
    pb["team_ortg"] = 0.0
    pb["team_drtg"] = 0.0
    pb["team_pace"] = 0.0
    pb["opp_ortg"]  = 0.0
    pb["opp_drtg"]  = 0.0
    pb["opp_pace"]  = 0.0

    if tb_date and tb_team:
        t = tb.rename(columns={tb_date:"game_date", tb_team:"team"}).copy()
        t["game_date"] = pd.to_datetime(t["game_date"], errors="coerce")
        t = t.dropna(subset=["game_date"])
        t["team"] = t["team"].astype(str)

        ren = {}
        if ortg: ren[ortg] = "ortg"
        if drtg: ren[drtg] = "drtg"
        if pace: ren[pace] = "pace"

        if ren:
            t = t.rename(columns=ren)
            keep = ["game_date","team"] + list(ren.values())
            t = t[keep].drop_duplicates(["game_date","team"])

            # merge team metrics
            tm = t.rename(columns={"ortg":"team_ortg","drtg":"team_drtg","pace":"team_pace"})
            pb = pb.merge(tm, on=["game_date","team"], how="left")

            # merge opponent metrics (only works if pb["opp"] matches tb "team" keys; if not, fills 0)
            om = t.rename(columns={"team":"opp","ortg":"opp_ortg","drtg":"opp_drtg","pace":"opp_pace"})
            pb = pb.merge(om, on=["game_date","opp"], how="left")

    # numeric cleanup
    for c in ["team_ortg","team_drtg","team_pace","opp_ortg","opp_drtg","opp_pace"]:
        pb[c] = pd.to_numeric(pb.get(c, 0.0), errors="coerce").fillna(0.0)

    # --- REST / B2B from team_box (big prop signal) ---
    rest_tbl = _team_rest_table(tb)
    pb["rest_days"] = 0.0
    pb["b2b"] = 0
    pb["games_last_7d"] = 0.0

    if rest_tbl is not None and len(rest_tbl) > 0:
        pb = pb.merge(rest_tbl, on=["game_date","team"], how="left")
        pb["rest_days"] = pd.to_numeric(pb["rest_days"], errors="coerce").fillna(0.0)
        pb["b2b"] = pd.to_numeric(pb["b2b"], errors="coerce").fillna(0).astype(int)
        pb["games_last_7d"] = pd.to_numeric(pb["games_last_7d"], errors="coerce").fillna(0.0)

    # ✅ Keep ONLY safe numeric + core identifiers (prevents parquet type errors)
    engineered = [
        "gp_last14",
        "team_ortg","team_drtg","team_pace",
        "opp_ortg","opp_drtg","opp_pace",
        "rest_days","b2b","games_last_7d",
    ] + [c for c in pb.columns if c.endswith(("_r5","_r10","_sd10"))]

    keep_cols = ["player","game_date","team","opp","min","pts","reb","ast"] + engineered
    pb = pb[keep_cols].copy()

    # fill any remaining NaNs
    for c in engineered + ["min","pts","reb","ast"]:
        pb[c] = pd.to_numeric(pb[c], errors="coerce").fillna(0.0)

    out = ART / "features_v1.parquet"
    pb.to_parquet(out, index=False)
    print("Saved features:", out, "rows", len(pb), "cols", pb.shape[1])

if __name__ == "__main__":
    build_features()
