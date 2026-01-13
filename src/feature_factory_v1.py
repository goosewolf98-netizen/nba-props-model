from pathlib import Path
import pandas as pd
import numpy as np

RAW = Path("data/raw")
ART = Path("artifacts")

def _read_csv(name: str) -> pd.DataFrame:
    p = RAW / name
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
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

def build_features():
    ART.mkdir(parents=True, exist_ok=True)

    pb = _read_csv("nba_player_box.csv")
    tb = _read_csv("nba_team_box.csv")

    # ---------- Player box standardization ----------
    player_c = _pick(pb, ["player", "player_name", "athlete_display_name", "name"])
    date_c   = _pick(pb, ["game_date", "date", "start_date", "game_datetime"])
    teamid_c = _pick(pb, ["team_id"])
    oppabbr_c= _pick(pb, ["opponent_team_abbreviation", "opponent_abbreviation", "opp_abbr"])

    min_c    = _pick(pb, ["minutes", "min", "mp"])
    pts_c    = _pick(pb, ["points", "pts"])
    reb_c    = _pick(pb, ["rebounds", "reb", "trb"])
    ast_c    = _pick(pb, ["assists", "ast"])

    if any(x is None for x in [player_c, date_c, teamid_c, oppabbr_c, min_c, pts_c, reb_c, ast_c]):
        raise ValueError(f"nba_player_box.csv missing required columns. Have: {list(pb.columns)[:80]}")

    pb = pb.rename(columns={
        player_c:"player", date_c:"game_date", teamid_c:"team_id", oppabbr_c:"opp_abbr",
        min_c:"min", pts_c:"pts", reb_c:"reb", ast_c:"ast"
    }).copy()

    pb["game_date"] = pd.to_datetime(pb["game_date"], errors="coerce")
    pb = pb.dropna(subset=["game_date"])
    pb["player"] = pb["player"].astype(str)
    pb["opp_abbr"] = pb["opp_abbr"].astype(str)

    for c in ["min","pts","reb","ast"]:
        pb[c] = pd.to_numeric(pb[c], errors="coerce").fillna(0.0)

    # map team_id -> team_abbr from team box
    tb_teamid = _pick(tb, ["team_id"])
    tb_abbr   = _pick(tb, ["team_abbreviation", "team_abbr"])
    tb_date   = _pick(tb, ["game_date", "date", "start_date", "game_datetime"])
    tb_gid    = _pick(tb, ["game_id"])

    if any(x is None for x in [tb_teamid, tb_abbr, tb_date, tb_gid]):
        raise ValueError(f"nba_team_box.csv missing team_id/abbr/date/game_id. Have: {list(tb.columns)[:80]}")

    id_to_abbr = (
        tb[[tb_teamid, tb_abbr]]
        .dropna()
        .astype({tb_teamid: "int64"}, errors="ignore")
        .drop_duplicates()
        .set_index(tb_teamid)[tb_abbr]
        .to_dict()
    )

    pb["team_abbr"] = pb["team_id"].map(id_to_abbr).fillna("").astype(str)

    # ---------- Rolling player features (no leakage) ----------
    pb = pb.sort_values(["player","game_date"])
    g = pb.groupby("player", group_keys=False)

    for s in ["min","pts","reb","ast"]:
        pb[f"{s}_r5"]   = g[s].shift(1).rolling(5, min_periods=1).mean()
        pb[f"{s}_r10"]  = g[s].shift(1).rolling(10, min_periods=1).mean()
        pb[f"{s}_sd10"] = g[s].shift(1).rolling(10, min_periods=2).std()

    pb["gp_last14"] = g["game_date"].shift(1).rolling(14, min_periods=1).count()
    pb = pb.fillna(0.0)

    # ---------- Build team advanced metrics from team box ----------
    t = tb.rename(columns={tb_date:"game_date", tb_gid:"game_id", tb_abbr:"team_abbr"}).copy()
    t["game_date"] = pd.to_datetime(t["game_date"], errors="coerce")
    t = t.dropna(subset=["game_date"])
    t["team_abbr"] = t["team_abbr"].astype(str)

    # required boxscore cols for possessions estimate
    fga = _pick(t, ["field_goals_attempted", "fga"])
    fta = _pick(t, ["free_throws_attempted", "fta"])
    tov = _pick(t, ["turnovers", "tov"])
    orb = _pick(t, ["offensive_rebounds", "orb"])
    pts = _pick(t, ["team_score", "points", "pts"])

    # safe numeric
    for col in [fga, fta, tov, orb, pts]:
        if col is None:
            # if any missing, we just can’t compute advanced metrics; keep zeros
            t["poss"] = 0.0
            t["ortg_raw"] = 0.0
            t["pace_raw"] = 0.0
            break
    else:
        for col in [fga, fta, tov, orb, pts]:
            t[col] = pd.to_numeric(t[col], errors="coerce").fillna(0.0)

        # possessions approximation
        t["poss"] = (t[fga] + 0.44 * t[fta] - t[orb] + t[tov]).clip(lower=0.0)
        t["pace_raw"] = t["poss"]  # per-game estimate (good enough for now)
        t["ortg_raw"] = np.where(t["poss"] > 0, (t[pts] / t["poss"]) * 100.0, 0.0)

    # opponent join (same game_id, other team row)
    t2 = t[["game_id","team_abbr","poss","ortg_raw","pace_raw"]].copy()
    t2 = t2.rename(columns={
        "team_abbr":"opp_abbr",
        "poss":"opp_poss",
        "ortg_raw":"opp_ortg_raw",
        "pace_raw":"opp_pace_raw"
    })

    # join on game_id, exclude same team
    m = t.merge(t2, on="game_id", how="left")
    m = m[m["team_abbr"] != m["opp_abbr"]].copy()

    # defensive rating = opponent ORTG (same game possessions scale)
    m["drtg_raw"] = m["opp_ortg_raw"]

    # keep one opponent row per team-game
    m = m.sort_values(["game_id","team_abbr"]).drop_duplicates(["game_id","team_abbr"])

    # ---------- Rolling pregame team context (no leakage) ----------
    m = m.sort_values(["team_abbr","game_date"])
    tg = m.groupby("team_abbr", group_keys=False)

    m["team_ortg"] = tg["ortg_raw"].shift(1).rolling(10, min_periods=1).mean().fillna(0.0)
    m["team_drtg"] = tg["drtg_raw"].shift(1).rolling(10, min_periods=1).mean().fillna(0.0)
    m["team_pace"] = tg["pace_raw"].shift(1).rolling(10, min_periods=1).mean().fillna(0.0)

    # build opponent rolling tables by re-keying
    opp_roll = m[["game_date","team_abbr","team_ortg","team_drtg","team_pace"]].copy()
    opp_roll = opp_roll.rename(columns={
        "team_abbr":"opp_abbr",
        "team_ortg":"opp_ortg",
        "team_drtg":"opp_drtg",
        "team_pace":"opp_pace",
    })

    # ---------- Rest / B2B from team schedule (by team_abbr) ----------
    rest = m[["game_date","team_abbr"]].drop_duplicates().sort_values(["team_abbr","game_date"]).copy()
    rest["rest_days"] = rest.groupby("team_abbr")["game_date"].diff().dt.days
    rest["rest_days"] = rest["rest_days"].fillna(7).clip(lower=0, upper=14)
    rest["b2b"] = (rest["rest_days"] <= 1).astype(int)

    def games_last_7d(g):
        g = g.sort_values("game_date").copy()
        g = g.set_index("game_date")
        cnt = g["b2b"].rolling("7D").count().fillna(0) - 1
        g["games_last_7d"] = cnt.clip(lower=0).values
        return g.reset_index()

    rest = rest.groupby("team_abbr", group_keys=False).apply(games_last_7d)

    # ---------- Merge team context into player rows ----------
    pb = pb.merge(
        m[["game_date","team_abbr","team_ortg","team_drtg","team_pace"]],
        on=["game_date","team_abbr"], how="left"
    )
    pb = pb.merge(
        opp_roll,
        on=["game_date","opp_abbr"], how="left"
    )
    pb = pb.merge(
        rest[["game_date","team_abbr","rest_days","b2b","games_last_7d"]],
        on=["game_date","team_abbr"], how="left"
    )

    # fill defaults
    for c in ["team_ortg","team_drtg","team_pace","opp_ortg","opp_drtg","opp_pace","rest_days","games_last_7d"]:
        if c not in pb.columns:
            pb[c] = 0.0
        pb[c] = pd.to_numeric(pb[c], errors="coerce").fillna(0.0)
    if "b2b" not in pb.columns:
        pb["b2b"] = 0
    pb["b2b"] = pd.to_numeric(pb["b2b"], errors="coerce").fillna(0).astype(int)

    # output
    engineered = [
        "gp_last14",
        "team_ortg","team_drtg","team_pace",
        "opp_ortg","opp_drtg","opp_pace",
        "rest_days","b2b","games_last_7d",
    ] + [c for c in pb.columns if c.endswith(("_r5","_r10","_sd10"))]

    keep_cols = ["player","game_date","team_abbr","opp_abbr","min","pts","reb","ast"] + engineered
    out_df = pb[keep_cols].copy()
    for c in engineered + ["min","pts","reb","ast"]:
        out_df[c] = pd.to_numeric(out_df[c], errors="coerce").fillna(0.0)

    out = ART / "features_v1.parquet"
    out_df.to_parquet(out, index=False)
    print("Saved features:", out, "rows", len(out_df), "cols", out_df.shape[1])

if __name__ == "__main__":
    build_features()
