from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from schema_normalize import norm_all


TEAM_BOX = Path("data/raw/nba_team_box.csv")
PLAYER_BOX = Path("data/raw/nba_player_box.csv")
OUT_PATH = Path("artifacts/opponent_matchup_features.parquet")


def _load_team_box() -> pd.DataFrame:
    if not TEAM_BOX.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(TEAM_BOX)
    except Exception:
        return pd.DataFrame()
    df = norm_all(df)
    return df


def _load_player_box() -> pd.DataFrame:
    if not PLAYER_BOX.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(PLAYER_BOX)
    except Exception:
        return pd.DataFrame()
    df = norm_all(df)
    return df


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tb = _load_team_box()
    pb = _load_player_box()
    if tb.empty or pb.empty:
        pd.DataFrame(columns=[
            "game_date", "opp_abbr",
            "opp_def_rating_roll", "opp_pace_roll",
            "opp_reb_rate_allowed", "opp_ast_rate_allowed",
        ]).to_parquet(OUT_PATH, index=False)
        print("Missing team/player box; wrote empty opponent matchup features.")
        return

    for col in ["game_date", "team_abbr"]:
        if col not in tb.columns:
            tb[col] = ""

    tb["game_date"] = pd.to_datetime(tb["game_date"], errors="coerce")
    tb = tb.dropna(subset=["game_date"])
    tb = tb.sort_values(["team_abbr", "game_date"])

    fga = next((c for c in ["field_goals_attempted", "fga"] if c in tb.columns), None)
    fta = next((c for c in ["free_throws_attempted", "fta"] if c in tb.columns), None)
    tov = next((c for c in ["turnovers", "tov"] if c in tb.columns), None)
    orb = next((c for c in ["offensive_rebounds", "orb"] if c in tb.columns), None)
    pts = next((c for c in ["team_score", "points", "pts"] if c in tb.columns), None)

    if all([fga, fta, tov, orb, pts]):
        for col in [fga, fta, tov, orb, pts]:
            tb[col] = pd.to_numeric(tb[col], errors="coerce").fillna(0.0)
        tb["poss"] = (tb[fga] + 0.44 * tb[fta] - tb[orb] + tb[tov]).clip(lower=0.0)
        tb["pace_raw"] = tb["poss"]
        tb["drtg_raw"] = np.where(tb["poss"] > 0, (tb[pts] / tb["poss"]) * 100.0, 0.0)
    else:
        tb["pace_raw"] = 0.0
        tb["drtg_raw"] = 0.0

    tb["opp_def_rating_roll"] = (
        tb.groupby("team_abbr")["drtg_raw"].shift(1).rolling(10, min_periods=1).mean().fillna(0.0)
    )
    tb["opp_pace_roll"] = (
        tb.groupby("team_abbr")["pace_raw"].shift(1).rolling(10, min_periods=1).mean().fillna(0.0)
    )

    opp_roll = tb[["game_date", "team_abbr", "opp_def_rating_roll", "opp_pace_roll"]].copy()
    opp_roll = opp_roll.rename(columns={"team_abbr": "opp_abbr"})

    pb = norm_all(pb)
    pb = pb.reindex(columns=["game_date", "opp_abbr"])
    pb = pb.dropna(subset=["game_date"]).drop_duplicates()
    merged = pb.merge(opp_roll, on=["game_date", "opp_abbr"], how="left")
    merged["opp_reb_rate_allowed"] = 0.0
    merged["opp_ast_rate_allowed"] = 0.0
    merged = merged.fillna(0.0)

    merged.to_parquet(OUT_PATH, index=False)
    print("Saved opponent matchup features to", OUT_PATH)


if __name__ == "__main__":
    main()
