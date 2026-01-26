from __future__ import annotations

from pathlib import Path
import pandas as pd

from schema_normalize import norm_all, norm_minutes


RAW_PLAYER = Path("data/raw/nba_player_box.csv")
AVAIL_PATH = Path("data/injuries/availability_by_game.csv")
OUT_PATH = Path("artifacts/with_without_features.parquet")


def _load_player_box() -> pd.DataFrame:
    if not RAW_PLAYER.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(RAW_PLAYER)
    except Exception:
        return pd.DataFrame()
    df = norm_all(df)
    return df


def _load_availability() -> pd.DataFrame:
    if not AVAIL_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(AVAIL_PATH)
    except Exception:
        return pd.DataFrame()


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pb = _load_player_box()
    if pb.empty:
        pd.DataFrame(columns=[
            "game_date", "team_abbr", "player",
            "starters_out_count", "rotation_out_count",
            "top_usage_teammates_out_count",
            "top_ast_teammates_out_count",
            "top_reb_teammates_out_count",
        ]).to_parquet(OUT_PATH, index=False)
        print("Player box missing; wrote empty with/without features.")
        return

    player_col = "player"
    pb = norm_minutes(pb)
    min_col = "min" if "min" in pb.columns else ("minutes" if "minutes" in pb.columns else None)
    if min_col is None:
        pb["min"] = 0.0
        min_col = "min"
    for col in [player_col, min_col, "team_abbr", "game_date"]:
        if col not in pb.columns:
            pb[col] = ""

    pb[min_col] = pd.to_numeric(pb[min_col], errors="coerce").fillna(0.0)
    pb["player"] = pb[player_col].astype(str)
    pb["team_abbr"] = pb["team_abbr"].astype(str)
    pb["game_date"] = pb["game_date"].astype(str)

    avail = _load_availability()
    if avail.empty:
        avail = pd.DataFrame(columns=["game_date", "team_abbr", "player", "is_out"])
    for col in ["game_date", "team_abbr", "player", "is_out"]:
        if col not in avail.columns:
            avail[col] = "" if col != "is_out" else 0
    avail["is_out"] = pd.to_numeric(avail["is_out"], errors="coerce").fillna(0).astype(int)

    team_usage = pb.groupby(["team_abbr", "player"], as_index=False).agg(
        pts=("pts", "mean") if "pts" in pb.columns else ("min", "mean"),
        reb=("reb", "mean") if "reb" in pb.columns else ("min", "mean"),
        ast=("ast", "mean") if "ast" in pb.columns else ("min", "mean"),
    )
    top_usage = team_usage.groupby("team_abbr").apply(
        lambda g: g.sort_values("pts", ascending=False).head(3)
    ).reset_index(drop=True)
    top_ast = team_usage.groupby("team_abbr").apply(
        lambda g: g.sort_values("ast", ascending=False).head(3)
    ).reset_index(drop=True)
    top_reb = team_usage.groupby("team_abbr").apply(
        lambda g: g.sort_values("reb", ascending=False).head(3)
    ).reset_index(drop=True)

    top_usage_map = top_usage.groupby("team_abbr")["player"].apply(list).to_dict()
    top_ast_map = top_ast.groupby("team_abbr")["player"].apply(list).to_dict()
    top_reb_map = top_reb.groupby("team_abbr")["player"].apply(list).to_dict()

    out_rows = []
    for _, row in pb.iterrows():
        team = row.get("team_abbr", "")
        game_date = row.get("game_date", "")
        player = row.get("player", "")
        out_df = avail[(avail["team_abbr"] == team) & (avail["game_date"] == game_date) & (avail["is_out"] == 1)]
        out_players = set(out_df["player"].astype(str).tolist())

        starters_out = len(out_players)
        rotation_out = len(out_players)
        top_usage_out = len([p for p in top_usage_map.get(team, []) if p in out_players and p != player])
        top_ast_out = len([p for p in top_ast_map.get(team, []) if p in out_players and p != player])
        top_reb_out = len([p for p in top_reb_map.get(team, []) if p in out_players and p != player])

        out_rows.append({
            "game_date": game_date,
            "team_abbr": team,
            "player": player,
            "starters_out_count": starters_out,
            "rotation_out_count": rotation_out,
            "top_usage_teammates_out_count": top_usage_out,
            "top_ast_teammates_out_count": top_ast_out,
            "top_reb_teammates_out_count": top_reb_out,
        })

    out_df = pd.DataFrame(out_rows)
    out_df = out_df.drop_duplicates(subset=["game_date", "team_abbr", "player"])
    out_df.to_parquet(OUT_PATH, index=False)
    print("Saved with/without features to", OUT_PATH)


if __name__ == "__main__":
    main()
