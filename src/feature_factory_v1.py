from pathlib import Path
import os
import subprocess
import pandas as pd
import numpy as np

from availability_features import add_availability_features
from build_lineup_cache import ensure_lineup_cache
from injuries_official import fetch_latest_injuries
from schema_normalize import ensure_col, normalize_dates, safe_cols, norm_all, norm_game_date

RAW = Path("data/raw")
ART = Path("artifacts")

def _norm_game_date(df):
    if df is None or len(df) == 0:
        return df
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date.astype(str)
    elif "date" in df.columns:
        df["game_date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
    return df


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

    try:
        fetch_latest_injuries(ART / "injuries_latest.csv")
    except Exception as exc:
        print(f"Injury ingest skipped: {exc}")
    try:
        ensure_lineup_cache()
    except Exception as exc:
        print(f"Lineup cache skipped: {exc}")
    try:
        subprocess.run(["python", "src/sdi_injuries.py"], check=False)
        subprocess.run(["python", "src/build_availability_table.py"], check=False)
        subprocess.run(["python", "src/with_without.py"], check=False)
        subprocess.run(["python", "src/opponent_matchup.py"], check=False)
    except Exception as exc:
        print(f"Context builders skipped: {exc}")

    pb = _read_csv("nba_player_box.csv")
    pb = _norm_game_date(pb)
    pb = norm_all(pb)
    pb = normalize_dates(pb)
    pb = ensure_col(pb, "team_abbr", ["team_abbreviation", "team", "abbr", "TEAM", "TEAM_ABBR", "TeamAbbr"])
    pb = ensure_col(pb, "opp_abbr", ["opp_abbreviation", "opp", "opponent", "OPP", "OPP_ABBR", "OpponentAbbr"])
    if "team_abbr" not in pb.columns:
        pb["team_abbr"] = ""
    if "opp_abbr" not in pb.columns:
        pb["opp_abbr"] = ""
    print("PLAYER cols:", list(pb.columns))

    tb = _read_csv("nba_team_box.csv")
    tb = _norm_game_date(tb)
    tb = norm_all(tb)
    tb = normalize_dates(tb)
    tb = ensure_col(tb, "team_abbr", ["team_abbreviation", "team", "abbr", "TEAM", "TEAM_ABBR", "TeamAbbr"])
    tb = ensure_col(tb, "opp_abbr", ["opp_abbreviation", "opp", "opponent", "OPP", "OPP_ABBR", "OpponentAbbr"])
    print("TEAM cols:", list(tb.columns))

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
        print(
            "Warning: nba_player_box.csv missing required columns; "
            f"have: {list(pb.columns)[:80]}"
        )
        pb = pd.DataFrame(
            columns=["player", "game_date", "team_id", "team_abbr", "opp_abbr", "min", "pts", "reb", "ast"]
        )
    else:
        pb = pb.rename(columns={
            player_c:"player", date_c:"game_date", teamid_c:"team_id", oppabbr_c:"opp_abbr",
            min_c:"min", pts_c:"pts", reb_c:"reb", ast_c:"ast"
        }).copy()

    pb = _norm_game_date(pb)
    pb = norm_all(pb)
    pb = normalize_dates(pb)
    pb = ensure_col(pb, "team_abbr", ["team_abbreviation", "team", "abbr", "TEAM", "TEAM_ABBR", "TeamAbbr"])
    pb = ensure_col(pb, "opp_abbr", ["opp_abbreviation", "opp", "opponent", "OPP", "OPP_ABBR", "OpponentAbbr"])
    if any(c not in pb.columns for c in ["game_date", "player"]):
        print("Warning: player_df missing required columns; writing zeros.")
        pb = pd.DataFrame(
            columns=["player", "game_date", "team_id", "team_abbr", "opp_abbr", "min", "pts", "reb", "ast"]
        )

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
        print(
            "Warning: nba_team_box.csv missing team_id/abbr/date/game_id; "
            f"have: {list(tb.columns)[:80]}"
        )
        tb_teamid = None
        tb_abbr = None
        tb_date = None
        tb_gid = None

    if tb_teamid and tb_abbr:
        id_to_abbr = (
            safe_cols(tb, [tb_teamid, tb_abbr])
            .dropna()
            .astype({tb_teamid: "int64"}, errors="ignore")
            .drop_duplicates()
            .set_index(tb_teamid)[tb_abbr]
            .to_dict()
        )
    else:
        id_to_abbr = {}

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
    if tb_date and tb_gid and tb_abbr:
        t = tb.rename(columns={tb_date:"game_date", tb_gid:"game_id", tb_abbr:"team_abbr"}).copy()
    else:
        t = pd.DataFrame(columns=["game_date", "game_id", "team_abbr"])
    t = _norm_game_date(t)
    t = norm_all(t)
    t = normalize_dates(t)
    t = ensure_col(t, "team_abbr", ["team_abbreviation", "team", "abbr", "TEAM", "TEAM_ABBR", "TeamAbbr"])
    t = ensure_col(t, "opp_abbr", ["opp_abbreviation", "opp", "opponent", "OPP", "OPP_ABBR", "OpponentAbbr"])
    if any(c not in t.columns for c in ["game_date", "team_abbr"]):
        print("Warning: team_df missing required columns; writing zeros.")
        t = pd.DataFrame(columns=["game_date", "game_id", "team_abbr"])
        t["game_date"] = pd.to_datetime(t["game_date"], errors="coerce")
    if "team_abbr" not in t.columns:
        t["team_abbr"] = ""
    t["game_date"] = pd.to_datetime(t["game_date"], errors="coerce")
    t = t.dropna(subset=["game_date"])
    t["team_abbr"] = t["team_abbr"].astype(str)
    print("SCHED cols:", list(t.columns))

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
    t2 = safe_cols(
        t,
        ["game_id","team_abbr","poss","ortg_raw","pace_raw"],
        fill_zero_cols=["poss", "ortg_raw", "pace_raw"],
    ).copy()
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

    m["team_ortg_roll10"] = tg["ortg_raw"].shift(1).rolling(10, min_periods=1).mean().fillna(0.0)
    m["team_drtg_roll10"] = tg["drtg_raw"].shift(1).rolling(10, min_periods=1).mean().fillna(0.0)
    m["team_pace_roll10"] = tg["pace_raw"].shift(1).rolling(10, min_periods=1).mean().fillna(0.0)

    # backwards-compatible names
    m["team_ortg"] = m["team_ortg_roll10"]
    m["team_drtg"] = m["team_drtg_roll10"]
    m["team_pace"] = m["team_pace_roll10"]

    # build opponent rolling tables by re-keying
    opp_roll = safe_cols(
        m,
        ["game_date","team_abbr","team_ortg_roll10","team_drtg_roll10","team_pace_roll10"],
        fill_zero_cols=["team_ortg_roll10", "team_drtg_roll10", "team_pace_roll10"],
    ).copy()
    opp_roll = _norm_game_date(opp_roll)
    opp_roll = norm_all(opp_roll)
    opp_roll = opp_roll.rename(columns={
        "team_abbr":"opp_abbr",
        "team_ortg_roll10":"opp_ortg_roll10",
        "team_drtg_roll10":"opp_drtg_roll10",
        "team_pace_roll10":"opp_pace_roll10",
    })

    opp_roll["opp_ortg"] = opp_roll["opp_ortg_roll10"]
    opp_roll["opp_drtg"] = opp_roll["opp_drtg_roll10"]
    opp_roll["opp_pace"] = opp_roll["opp_pace_roll10"]

    # ---------- Rest / B2B from team schedule (by team_abbr) ----------
    rest = _norm_game_date(m)
    rest = norm_all(m)
    rest = normalize_dates(rest)
    rest = ensure_col(rest, "team_abbr", ["team_abbreviation", "team", "abbr", "TEAM", "TEAM_ABBR", "TeamAbbr"])
    rest = ensure_col(rest, "opp_abbr", ["opp_abbreviation", "opp", "opponent", "OPP", "OPP_ABBR", "OpponentAbbr"])
    print("rest_df columns:", list(rest.columns))
    rest = safe_cols(rest, ["game_date", "team_abbr"])
    rest["game_date"] = pd.to_datetime(rest["game_date"], errors="coerce")
    if any(c not in rest.columns for c in ["game_date", "team_abbr"]) or not rest["team_abbr"].notna().any():
        print("Warning: rest_df missing required columns; writing zeros.")
        rest = pd.DataFrame(columns=["game_date","team_abbr","rest_days","b2b","games_last_7d"])
    if "team_abbr" in rest.columns and rest["team_abbr"].notna().any():
        rest = rest.drop_duplicates().sort_values(["team_abbr","game_date"]).copy()
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
        rest = safe_cols(
            rest,
            ["game_date","team_abbr","rest_days","b2b","games_last_7d"],
            fill_zero_cols=["rest_days", "b2b", "games_last_7d"],
        )
    else:
        rest = pd.DataFrame(columns=["game_date","team_abbr","rest_days","b2b","games_last_7d"])

    # ---------- Merge team context into player rows ----------
    pb = _norm_game_date(pb)
    m = _norm_game_date(m)
    pb = norm_all(pb)
    m = norm_all(m)
    print("MERGE game_date dtypes:", pb["game_date"].dtype, m["game_date"].dtype)
    pb = pb.merge(
        safe_cols(
            m,
            ["game_date","team_abbr","team_ortg","team_drtg","team_pace",
             "team_ortg_roll10","team_drtg_roll10","team_pace_roll10"],
            fill_zero_cols=[
                "team_ortg",
                "team_drtg",
                "team_pace",
                "team_ortg_roll10",
                "team_drtg_roll10",
                "team_pace_roll10",
            ],
        ),
        on=["game_date","team_abbr"], how="left"
    )
    pb = _norm_game_date(pb)
    opp_roll = _norm_game_date(opp_roll)
    pb = norm_all(pb)
    opp_roll = norm_all(opp_roll)
    print("MERGE game_date dtypes:", pb["game_date"].dtype, opp_roll["game_date"].dtype)
    pb = pb.merge(
        opp_roll,
        on=["game_date","opp_abbr"], how="left"
    )
    if "team_abbr" not in pb.columns and "team_abbreviation" in pb.columns:
        pb = pb.rename(columns={"team_abbreviation": "team_abbr"})
    rest = ensure_col(rest, "team_abbr", ["team_abbreviation", "team", "abbr", "TEAM", "TEAM_ABBR", "TeamAbbr"])
    if "team_abbr" not in rest.columns:
        if "team_abbreviation" in rest.columns:
            rest = rest.rename(columns={"team_abbreviation": "team_abbr"})
        elif "team" in rest.columns:
            rest = rest.rename(columns={"team": "team_abbr"})
        elif "abbr" in rest.columns:
            rest = rest.rename(columns={"abbr": "team_abbr"})
    rest = rest.reindex(columns=["game_date","team_abbr","rest_days","b2b","games_last_7d"])
    for c in ["rest_days", "b2b", "games_last_7d"]:
        if c in rest.columns:
            rest[c] = rest[c].fillna(0)
    print("REST COLS:", list(rest.columns))
    print("REST team_abbr present:", "team_abbr" in rest.columns)
    print("MERGE game_date dtypes:", pb["game_date"].dtype, rest["game_date"].dtype)
    if "team_abbr" in pb.columns and "team_abbr" in rest.columns:
        pb = pb.merge(
            rest,
            on=["game_date","team_abbr"], how="left"
        )
        for col in ["rest_days", "b2b", "games_last_7d"]:
            pb[col] = pb[col].fillna(0)
    else:
        pb["rest_days"] = 0.0
        pb["b2b"] = 0
        pb["games_last_7d"] = 0.0

    pb = add_availability_features(pb, ART / "injuries_latest.csv")

    availability_path = Path("data/injuries/availability_by_game.csv")
    if availability_path.exists():
        try:
            availability = pd.read_csv(availability_path)
        except Exception:
            availability = pd.DataFrame()
    else:
        availability = pd.DataFrame()

    availability = norm_all(availability)
    if not availability.empty:
        for col in ["game_date", "team_abbr", "player", "is_out", "is_doubt", "is_q", "is_prob", "is_active"]:
            if col not in availability.columns:
                availability[col] = 0 if col.startswith("is_") else ""
        availability["game_date"] = pd.to_datetime(availability["game_date"], errors="coerce").dt.date.astype(str)
        availability["team_abbr"] = availability["team_abbr"].astype(str)
        team_avail = availability.groupby(["game_date", "team_abbr"], as_index=False).agg(
            team_out_count=("is_out", "sum"),
            team_doubt_count=("is_doubt", "sum"),
            team_q_count=("is_q", "sum"),
            team_prob_count=("is_prob", "sum"),
        )
        opp_avail = team_avail.rename(columns={
            "team_abbr": "opp_abbr",
            "team_out_count": "opp_out_count",
            "team_doubt_count": "opp_doubt_count",
            "team_q_count": "opp_q_count",
            "team_prob_count": "opp_prob_count",
        })
        pb["game_date_key"] = pb["game_date"].dt.date.astype(str)
        pb = pb.merge(team_avail, left_on=["game_date_key", "team_abbr"], right_on=["game_date", "team_abbr"], how="left")
        pb = pb.merge(opp_avail, left_on=["game_date_key", "opp_abbr"], right_on=["game_date", "opp_abbr"], how="left")
        pb = pb.drop(columns=["game_date_key", "game_date_x", "game_date_y"], errors="ignore")
    else:
        pb["team_out_count"] = pb.get("team_out_count", 0.0)
        pb["team_doubt_count"] = 0.0
        pb["team_q_count"] = pb.get("team_q_count", 0.0)
        pb["team_prob_count"] = 0.0
        pb["opp_out_count"] = pb.get("opp_out_count", 0.0)
        pb["opp_doubt_count"] = 0.0
        pb["opp_q_count"] = pb.get("opp_q_count", 0.0)
        pb["opp_prob_count"] = 0.0

    with_without_path = ART / "with_without_features.parquet"
    if with_without_path.exists():
        try:
            with_without = pd.read_parquet(with_without_path)
        except Exception:
            with_without = pd.DataFrame()
        if not with_without.empty:
            with_without = norm_all(with_without)
            for col in ["game_date", "team_abbr", "player"]:
                if col not in with_without.columns:
                    with_without[col] = ""
            pb = pb.merge(
                with_without,
                on=["game_date", "team_abbr", "player"],
                how="left",
            )

    opponent_matchup_path = ART / "opponent_matchup_features.parquet"
    if opponent_matchup_path.exists():
        try:
            opp_match = pd.read_parquet(opponent_matchup_path)
        except Exception:
            opp_match = pd.DataFrame()
        if not opp_match.empty:
            opp_match = norm_all(opp_match)
            for col in ["game_date", "opp_abbr"]:
                if col not in opp_match.columns:
                    opp_match[col] = ""
            pb = pb.merge(
                opp_match,
                on=["game_date", "opp_abbr"],
                how="left",
            )

    use_market_open = os.getenv("USE_MARKET_OPEN_FEATURES", "0") == "1"
    line_moves_path = Path("data/lines/props_line_movement.csv")
    for stat in ["pts", "reb", "ast"]:
        pb[f"market_open_line_{stat}"] = 0.0
        pb[f"market_open_implied_over_{stat}"] = 0.0
        pb[f"market_open_implied_under_{stat}"] = 0.0
        pb[f"market_book_count_{stat}"] = 0.0
        pb[f"early_line_move_{stat}"] = 0.0

    if use_market_open and line_moves_path.exists():
        try:
            line_moves = pd.read_csv(line_moves_path)
        except Exception:
            line_moves = pd.DataFrame()
        if not line_moves.empty:
            for col in ["game_date", "player", "stat", "open_line", "open_over_odds", "open_under_odds", "book", "open_ts"]:
                if col not in line_moves.columns:
                    line_moves[col] = pd.NA
            line_moves["game_date"] = pd.to_datetime(line_moves["game_date"], errors="coerce").dt.date.astype(str)
            line_moves["player_norm"] = line_moves["player"].astype(str).str.lower().str.split().str.join(" ")
            line_moves["stat_norm"] = line_moves["stat"].astype(str).str.lower()
            line_moves["open_line"] = pd.to_numeric(line_moves["open_line"], errors="coerce")
            line_moves["open_over_odds"] = pd.to_numeric(line_moves["open_over_odds"], errors="coerce")
            line_moves["open_under_odds"] = pd.to_numeric(line_moves["open_under_odds"], errors="coerce")
            line_moves["open_ts"] = pd.to_datetime(line_moves["open_ts"], errors="coerce")
            line_moves = line_moves.sort_values("open_ts")
            line_moves["early_line"] = line_moves.groupby(
                ["game_date", "player_norm", "stat_norm", "book"], dropna=False
            )["open_line"].transform(lambda s: s.iloc[1] if len(s) > 1 else s.iloc[0])
            line_moves["early_line_move"] = (line_moves["early_line"] - line_moves["open_line"]).fillna(0.0)

            def implied_prob(odds):
                if pd.isna(odds):
                    return 0.0
                odds = float(odds)
                return abs(odds) / (abs(odds) + 100.0) if odds < 0 else 100.0 / (odds + 100.0)

            line_moves["implied_over"] = line_moves["open_over_odds"].apply(implied_prob)
            line_moves["implied_under"] = line_moves["open_under_odds"].apply(implied_prob)
            agg = line_moves.groupby(["game_date", "player_norm", "stat_norm"], as_index=False).agg(
                market_open_line=("open_line", "mean"),
                market_open_implied_over=("implied_over", "mean"),
                market_open_implied_under=("implied_under", "mean"),
                market_book_count=("book", "nunique"),
                early_line_move=("early_line_move", "mean"),
            )
            pb["player_norm"] = pb["player"].astype(str).str.lower().str.split().str.join(" ")
            pb["game_date_key"] = pb["game_date"].dt.date.astype(str)
            for stat in ["pts", "reb", "ast"]:
                stat_rows = agg[agg["stat_norm"] == stat].copy()
                if stat_rows.empty:
                    continue
                stat_rows = stat_rows.rename(columns={
                    "market_open_line": f"market_open_line_{stat}",
                    "market_open_implied_over": f"market_open_implied_over_{stat}",
                    "market_open_implied_under": f"market_open_implied_under_{stat}",
                    "market_book_count": f"market_book_count_{stat}",
                    "early_line_move": f"early_line_move_{stat}",
                })
                pb = pb.merge(
                    safe_cols(
                        stat_rows,
                        [
                            "game_date", "player_norm",
                            f"market_open_line_{stat}",
                            f"market_open_implied_over_{stat}",
                            f"market_open_implied_under_{stat}",
                            f"market_book_count_{stat}",
                            f"early_line_move_{stat}",
                        ],
                        fill_zero_cols=[
                            f"market_open_line_{stat}",
                            f"market_open_implied_over_{stat}",
                            f"market_open_implied_under_{stat}",
                            f"market_book_count_{stat}",
                            f"early_line_move_{stat}",
                        ],
                    ).rename(columns={"game_date": "game_date_key"}),
                    on=["game_date_key", "player_norm"],
                    how="left",
                )
            pb = pb.drop(columns=["player_norm", "game_date_key"], errors="ignore")

    lineup_player_path = Path("data/derived/player_onoff_cache.parquet")
    lineup_team_path = Path("data/derived/team_onoff_cache.parquet")
    lineup_cache_timestamp = None

    if lineup_player_path.exists():
        try:
            player_onoff = pd.read_parquet(lineup_player_path)
        except Exception:
            player_onoff = pd.DataFrame()
        if len(player_onoff) > 0:
            for col in ["player", "team_abbr"]:
                if col not in player_onoff.columns:
                    player_onoff[col] = ""
            for col in ["player_on_off_net", "player_on_off_pace", "player_minutes_on_recent", "minutes_on", "minutes_off"]:
                if col not in player_onoff.columns:
                    player_onoff[col] = 0.0
                player_onoff[col] = pd.to_numeric(player_onoff[col], errors="coerce").fillna(0.0)
            if "cache_timestamp" in player_onoff.columns:
                lineup_cache_timestamp = str(player_onoff["cache_timestamp"].iloc[0])
            pb = pb.merge(
                safe_cols(
                    player_onoff,
                    [
                        "player", "team_abbr",
                        "player_on_off_net", "player_on_off_pace", "player_minutes_on_recent",
                        "minutes_on", "minutes_off",
                    ],
                    fill_zero_cols=[
                        "player_on_off_net",
                        "player_on_off_pace",
                        "player_minutes_on_recent",
                        "minutes_on",
                        "minutes_off",
                    ],
                ),
                on=["player", "team_abbr"],
                how="left",
            )

    if lineup_team_path.exists():
        try:
            team_onoff = pd.read_parquet(lineup_team_path)
        except Exception:
            team_onoff = pd.DataFrame()
        if len(team_onoff) > 0:
            for col in ["team_abbr", "team_def_net_recent"]:
                if col not in team_onoff.columns:
                    team_onoff[col] = 0.0
            team_onoff["team_def_net_recent"] = pd.to_numeric(team_onoff["team_def_net_recent"], errors="coerce").fillna(0.0)
            if "cache_timestamp" in team_onoff.columns and lineup_cache_timestamp is None:
                lineup_cache_timestamp = str(team_onoff["cache_timestamp"].iloc[0])
            pb = pb.merge(
                safe_cols(
                    team_onoff,
                    ["team_abbr", "team_def_net_recent"],
                    fill_zero_cols=["team_def_net_recent"],
                ).rename(columns={"team_abbr": "opp_abbr", "team_def_net_recent": "opp_def_net_recent"}),
                on="opp_abbr",
                how="left",
            )

    if "player_on_off_net" not in pb.columns:
        pb["player_on_off_net"] = 0.0
    if "player_on_off_pace" not in pb.columns:
        pb["player_on_off_pace"] = 0.0
    if "player_minutes_on_recent" not in pb.columns:
        pb["player_minutes_on_recent"] = 0.0
    if "opp_def_net_recent" not in pb.columns:
        pb["opp_def_net_recent"] = 0.0

    if lineup_cache_timestamp is None:
        pb["lineup_cache_timestamp"] = ""
    else:
        pb["lineup_cache_timestamp"] = lineup_cache_timestamp

    injuries_path = ART / "injuries_latest.csv"
    out_teammates = {}
    if injuries_path.exists():
        try:
            injuries_df = pd.read_csv(injuries_path)
        except Exception:
            injuries_df = pd.DataFrame()
        if len(injuries_df) > 0:
            for col in ["team_abbr", "player", "status"]:
                if col not in injuries_df.columns:
                    injuries_df[col] = ""
            injuries_df["team_abbr"] = injuries_df["team_abbr"].astype(str).str.upper()
            injuries_df["player"] = injuries_df["player"].astype(str)
            injuries_df["status"] = injuries_df["status"].astype(str).str.upper()
            out_df = injuries_df[injuries_df["status"].isin(["OUT", "DOUBTFUL"])]
            out_teammates = out_df.groupby("team_abbr")["player"].apply(lambda s: set(s.tolist())).to_dict()

    if "minutes_on" in pb.columns:
        teammate_rank = pb.groupby(["team_abbr", "player"], as_index=False)["minutes_on"].mean()
        teammate_rank = teammate_rank.sort_values(["team_abbr", "minutes_on"], ascending=[True, False])
        top_teammates = teammate_rank.groupby("team_abbr", as_index=False).head(3)
        top_map = top_teammates.groupby("team_abbr")["player"].apply(list).to_dict()
        def teammate_flags(row):
            team = row.get("team_abbr", "")
            player = row.get("player", "")
            top_list = [p for p in top_map.get(team, []) if p != player]
            out_set = out_teammates.get(team, set())
            if not top_list:
                return 0.0, 0.0
            out_flag = 1.0 if any(p in out_set for p in top_list) else 0.0
            on_flag = 1.0 if any(p not in out_set for p in top_list) else 0.0
            return on_flag, out_flag
        flags = pb.apply(lambda r: teammate_flags(r), axis=1, result_type="expand")
        pb["top_synergy_teammate_on_flag"] = flags[0]
        top_out_flag = flags[1]
    else:
        pb["top_synergy_teammate_on_flag"] = 0.0
        top_out_flag = 0.0

    pb["synergy_delta_proxy"] = np.where(
        pd.to_numeric(top_out_flag, errors="coerce").fillna(0.0) > 0,
        -pd.to_numeric(pb["player_on_off_net"], errors="coerce").fillna(0.0).abs(),
        pd.to_numeric(pb["player_on_off_net"], errors="coerce").fillna(0.0),
    )

    # fill defaults
    for c in [
        "team_ortg","team_drtg","team_pace",
        "team_ortg_roll10","team_drtg_roll10","team_pace_roll10",
        "opp_ortg","opp_drtg","opp_pace",
        "opp_ortg_roll10","opp_drtg_roll10","opp_pace_roll10",
        "rest_days","games_last_7d",
        "team_out_count","team_doubt_count","team_q_count","team_prob_count",
        "opp_out_count","opp_doubt_count","opp_q_count","opp_prob_count",
        "top_teammate_out_flag","out_teammates_min_proxy",
        "starters_out_count","rotation_out_count",
        "top_usage_teammates_out_count","top_ast_teammates_out_count","top_reb_teammates_out_count",
        "player_on_off_net","player_on_off_pace","player_minutes_on_recent",
        "opp_def_net_recent","top_synergy_teammate_on_flag","synergy_delta_proxy",
        "opp_def_rating_roll","opp_pace_roll","opp_reb_rate_allowed","opp_ast_rate_allowed",
        "market_open_line_pts","market_open_implied_over_pts","market_open_implied_under_pts","market_book_count_pts","early_line_move_pts",
        "market_open_line_reb","market_open_implied_over_reb","market_open_implied_under_reb","market_book_count_reb","early_line_move_reb",
        "market_open_line_ast","market_open_implied_over_ast","market_open_implied_under_ast","market_book_count_ast","early_line_move_ast",
    ]:
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
        "team_ortg_roll10","team_drtg_roll10","team_pace_roll10",
        "opp_ortg","opp_drtg","opp_pace",
        "opp_ortg_roll10","opp_drtg_roll10","opp_pace_roll10",
        "rest_days","b2b","games_last_7d",
        "team_out_count","team_doubt_count","team_q_count","team_prob_count",
        "opp_out_count","opp_doubt_count","opp_q_count","opp_prob_count",
        "top_teammate_out_flag","out_teammates_min_proxy",
        "starters_out_count","rotation_out_count",
        "top_usage_teammates_out_count","top_ast_teammates_out_count","top_reb_teammates_out_count",
        "player_on_off_net","player_on_off_pace","player_minutes_on_recent",
        "opp_def_net_recent","top_synergy_teammate_on_flag","synergy_delta_proxy",
        "opp_def_rating_roll","opp_pace_roll","opp_reb_rate_allowed","opp_ast_rate_allowed",
        "market_open_line_pts","market_open_implied_over_pts","market_open_implied_under_pts","market_book_count_pts","early_line_move_pts",
        "market_open_line_reb","market_open_implied_over_reb","market_open_implied_under_reb","market_book_count_reb","early_line_move_reb",
        "market_open_line_ast","market_open_implied_over_ast","market_open_implied_under_ast","market_book_count_ast","early_line_move_ast",
    ] + [c for c in pb.columns if c.endswith(("_r5","_r10","_sd10"))]

    keep_cols = ["player","game_date","team_abbr","opp_abbr","min","pts","reb","ast"] + engineered
    if "lineup_cache_timestamp" in pb.columns:
        keep_cols.append("lineup_cache_timestamp")
    out_df = safe_cols(
        pb,
        keep_cols,
        fill_zero_cols=engineered + ["min", "pts", "reb", "ast"],
    ).copy()
    for c in engineered + ["min","pts","reb","ast"]:
        out_df[c] = pd.to_numeric(out_df[c], errors="coerce").fillna(0.0)
    for col in ["rest_days", "b2b", "games_last_7d", "team_pace", "opp_pace", "team_drtg", "opp_drtg"]:
        if col not in out_df.columns:
            out_df[col] = 0.0

    if len(out_df) > 0:
        team_drtg_pct = float((out_df["team_drtg_roll10"] > 0).mean() * 100.0)
        opp_drtg_pct = float((out_df["opp_drtg_roll10"] > 0).mean() * 100.0)
    else:
        team_drtg_pct = 0.0
        opp_drtg_pct = 0.0
    print(f"Non-zero team_drtg_roll10: {team_drtg_pct:.1f}% | opp_drtg_roll10: {opp_drtg_pct:.1f}%")
    if len(out_df) > 0:
        onoff_pct = float((out_df["player_on_off_net"] != 0).mean() * 100.0)
    else:
        onoff_pct = 0.0
    print(f"Non-zero player_on_off_net: {onoff_pct:.1f}%")

    rest_sample = out_df.loc[
        out_df["rest_days"] > 0,
        ["player","game_date","team_abbr","rest_days","b2b","games_last_7d"],
    ].head(5)
    if rest_sample.empty:
        print("Sample rest days: None")
    else:
        print("Sample rest days:\n", rest_sample.to_string(index=False))

    out = ART / "features_v1.parquet"
    out_df.to_parquet(out, index=False)
    print("Saved features:", out, "rows", len(out_df), "cols", out_df.shape[1])
    if "rest_days" in out_df.columns:
        print("features_v1 rows:", len(out_df), "nonzero rest_days:", float((out_df["rest_days"] > 0).mean()))
    else:
        print("features_v1 rows:", len(out_df), "nonzero rest_days: 0.0")

if __name__ == "__main__":
    build_features()
