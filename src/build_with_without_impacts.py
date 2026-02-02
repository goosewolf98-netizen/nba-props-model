from __future__ import annotations

from pathlib import Path

import pandas as pd


def _load_lineups(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _load_boxscores(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _norm_player(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().str.split().str.join(" ")


def _build_presence_from_lineups(lineups: pd.DataFrame) -> pd.DataFrame:
    if lineups.empty:
        return pd.DataFrame()
    roster_cols = [c for c in lineups.columns if c.startswith("player")]
    if not roster_cols:
        return pd.DataFrame()
    base = lineups[["game_date", "team_abbr"] + roster_cols].copy()
    base = base.dropna(subset=["game_date", "team_abbr"], how="any")
    stacked = base.melt(
        id_vars=["game_date", "team_abbr"],
        value_vars=roster_cols,
        value_name="player",
    )
    stacked = stacked.dropna(subset=["player"]).copy()
    stacked["player_norm"] = _norm_player(stacked["player"])
    stacked["game_key"] = stacked["game_date"].astype(str) + "|" + stacked["team_abbr"].astype(str)
    return stacked[["game_key", "team_abbr", "player", "player_norm"]].drop_duplicates()


def _prepare_player_games(boxscores: pd.DataFrame) -> pd.DataFrame:
    if boxscores.empty:
        return pd.DataFrame()
    boxscores = boxscores.copy()
    boxscores["game_date"] = pd.to_datetime(boxscores.get("game_date"), errors="coerce").dt.date.astype(str)
    for col in ["team_abbr", "player"]:
        if col not in boxscores.columns:
            boxscores[col] = ""
    boxscores["player_norm"] = _norm_player(boxscores["player"])
    boxscores["team_abbr"] = boxscores["team_abbr"].astype(str)
    boxscores["min"] = pd.to_numeric(boxscores.get("min", 0), errors="coerce").fillna(0.0)
    boxscores["pts"] = pd.to_numeric(boxscores.get("pts", 0), errors="coerce").fillna(0.0)
    boxscores["reb"] = pd.to_numeric(boxscores.get("reb", 0), errors="coerce").fillna(0.0)
    boxscores["ast"] = pd.to_numeric(boxscores.get("ast", 0), errors="coerce").fillna(0.0)
    boxscores["usage_proxy"] = pd.to_numeric(boxscores.get("usage_proxy", 0), errors="coerce").fillna(0.0)
    boxscores["game_key"] = boxscores["game_date"].astype(str) + "|" + boxscores["team_abbr"].astype(str)
    boxscores = boxscores[boxscores["min"] > 0].copy()
    if boxscores.empty:
        return boxscores
    boxscores["pts_pm"] = boxscores["pts"] / boxscores["min"].replace(0, pd.NA)
    boxscores["reb_pm"] = boxscores["reb"] / boxscores["min"].replace(0, pd.NA)
    boxscores["ast_pm"] = boxscores["ast"] / boxscores["min"].replace(0, pd.NA)
    return boxscores


def main() -> None:
    lineup_path = Path("data/raw/kaggle_pbp_lineups.parquet")
    boxscore_path = Path("data/raw/player_box_hist.parquet")
    output_path = Path("data/models/with_without_impacts.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lineups = _load_lineups(lineup_path)
    boxscores = _load_boxscores(boxscore_path)

    if boxscores.empty:
        empty = pd.DataFrame(
            columns=[
                "team_abbr",
                "player",
                "teammate",
                "n_with",
                "n_without",
                "d_min",
                "d_pts_pm",
                "d_reb_pm",
                "d_ast_pm",
                "d_usg",
            ]
        )
        empty.to_parquet(output_path, index=False)
        print("Missing player boxscores; wrote empty with_without_impacts.parquet")
        return

    player_games = _prepare_player_games(boxscores)
    if player_games.empty:
        player_games.to_parquet(output_path, index=False)
        print("No player games with minutes; wrote empty with_without_impacts.parquet")
        return

    presence = _build_presence_from_lineups(lineups)
    if presence.empty:
        presence = player_games[["game_key", "team_abbr", "player", "player_norm"]].drop_duplicates()

    impacts: list[dict] = []
    for team_abbr, team_df in player_games.groupby("team_abbr"):
        team_df = team_df.copy()
        team_df = team_df.sort_values("game_date")
        team_players = sorted(team_df["player_norm"].unique())
        stats = team_df.set_index(["player_norm", "game_key"])
        presence_team = presence[presence["team_abbr"] == team_abbr]
        presence_map = presence_team.groupby("player_norm")["game_key"].apply(set).to_dict()

        for player_norm in team_players:
            p_games = set(team_df.loc[team_df["player_norm"] == player_norm, "game_key"])
            if len(p_games) < 8:
                continue
            for teammate_norm in team_players:
                if teammate_norm == player_norm:
                    continue
                t_games = presence_map.get(teammate_norm, set())
                with_games = p_games.intersection(t_games)
                without_games = p_games.difference(t_games)
                if len(with_games) < 8 or len(without_games) < 8:
                    continue

                with_stats = stats.loc[(player_norm, list(with_games))]
                without_stats = stats.loc[(player_norm, list(without_games))]

                d_min = without_stats["min"].mean() - with_stats["min"].mean()
                d_pts = without_stats["pts_pm"].mean() - with_stats["pts_pm"].mean()
                d_reb = without_stats["reb_pm"].mean() - with_stats["reb_pm"].mean()
                d_ast = without_stats["ast_pm"].mean() - with_stats["ast_pm"].mean()

                # Use usage_rate if available, fallback to usage_proxy
                u_col = "usage_rate" if "usage_rate" in without_stats.columns else "usage_proxy"
                d_usg = without_stats[u_col].mean() - with_stats[u_col].mean()

                sample_player = team_df[team_df["player_norm"] == player_norm].iloc[0]
                sample_teammate = team_df[team_df["player_norm"] == teammate_norm].iloc[0]

                impacts.append(
                    {
                        "team_abbr": team_abbr,
                        "player": sample_player["player"],
                        "teammate": sample_teammate["player"],
                        "n_with": len(with_games),
                        "n_without": len(without_games),
                        "d_min": float(d_min),
                        "d_pts_pm": float(d_pts),
                        "d_reb_pm": float(d_reb),
                        "d_ast_pm": float(d_ast),
                        "d_usg": float(d_usg),
                    }
                )

    impacts_df = pd.DataFrame(impacts)
    if impacts_df.empty:
        impacts_df = pd.DataFrame(
            columns=[
                "team_abbr",
                "player",
                "teammate",
                "n_with",
                "n_without",
                "d_min",
                "d_pts_pm",
                "d_reb_pm",
                "d_ast_pm",
                "d_usg",
            ]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    impacts_df.to_parquet(output_path, index=False)
    print("Saved", output_path)


if __name__ == "__main__":
    main()
