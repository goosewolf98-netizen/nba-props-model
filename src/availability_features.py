from __future__ import annotations

from pathlib import Path
import pandas as pd


def _empty_injuries() -> pd.DataFrame:
    return pd.DataFrame(columns=["report_datetime", "team_abbr", "player", "status", "reason"])


def _load_injuries(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _empty_injuries()
    try:
        df = pd.read_csv(path)
    except Exception:
        return _empty_injuries()
    if "report_datetime" not in df.columns:
        if "report_date" in df.columns:
            df = df.rename(columns={"report_date": "report_datetime"})
        elif "game_date" in df.columns:
            df = df.rename(columns={"game_date": "report_datetime"})
        else:
            df["report_datetime"] = pd.NaT
    for col in ["team_abbr", "player", "status", "reason"]:
        if col not in df.columns:
            df[col] = ""
    df["report_datetime"] = pd.to_datetime(df["report_datetime"], errors="coerce")
    df["team_abbr"] = df["team_abbr"].astype(str).str.upper()
    df["player"] = df["player"].astype(str)
    df["status"] = df["status"].astype(str).str.upper()
    df["reason"] = df["reason"].astype(str)
    return df


def add_availability_features(features: pd.DataFrame, injuries_path: Path) -> pd.DataFrame:
    if features.empty:
        return features

    out = features.copy()
    injuries = _load_injuries(injuries_path)

    if injuries.empty or injuries["report_datetime"].isna().all():
        out["team_out_count"] = 0.0
        out["team_q_count"] = 0.0
        out["opp_out_count"] = 0.0
        out["opp_q_count"] = 0.0
        out["top_teammate_out_flag"] = 0.0
        out["out_teammates_min_proxy"] = 0.0
        return out

    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out["_game_date"] = out["game_date"].dt.date
    injuries = injuries.dropna(subset=["report_datetime"]).copy()
    injuries["report_date"] = injuries["report_datetime"].dt.date

    injuries["is_out"] = injuries["status"].isin(["OUT", "DOUBTFUL"]).astype(int)
    injuries["is_q"] = (injuries["status"] == "QUESTIONABLE").astype(int)

    team_counts = (
        injuries.groupby(["team_abbr", "report_date"], as_index=False)
        .agg(team_out_count=("is_out", "sum"), team_q_count=("is_q", "sum"))
    )
    opp_counts = team_counts.rename(columns={
        "team_abbr": "opp_abbr",
        "team_out_count": "opp_out_count",
        "team_q_count": "opp_q_count",
    })

    out = out.merge(
        team_counts,
        left_on=["team_abbr", "_game_date"],
        right_on=["team_abbr", "report_date"],
        how="left",
    ).drop(columns=["report_date"], errors="ignore")
    if "opp_abbr" in out.columns:
        out = out.merge(
            opp_counts,
            left_on=["opp_abbr", "_game_date"],
            right_on=["opp_abbr", "report_date"],
            how="left",
        ).drop(columns=["report_date"], errors="ignore")
    else:
        out["opp_out_count"] = 0.0
        out["opp_q_count"] = 0.0

    for c in ["team_out_count", "team_q_count", "opp_out_count", "opp_q_count"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    usage_cols = ["player", "team_abbr", "game_date", "pts_r10", "min_r10"]
    missing_usage = [c for c in usage_cols if c not in out.columns]
    if missing_usage:
        out["top_teammate_out_flag"] = 0.0
        out["out_teammates_min_proxy"] = 0.0
        return out

    usage = out[usage_cols + ["_game_date"]].copy()
    usage = usage.rename(columns={"_game_date": "game_date"})
    usage["pts_r10"] = pd.to_numeric(usage["pts_r10"], errors="coerce").fillna(0.0)
    usage["min_r10"] = pd.to_numeric(usage["min_r10"], errors="coerce").fillna(0.0)
    usage["points_per_min"] = usage["pts_r10"] / usage["min_r10"].replace(0.0, pd.NA)
    usage["points_per_min"] = usage["points_per_min"].fillna(0.0)

    usage_ranked = usage.sort_values(
        ["team_abbr", "game_date", "points_per_min"],
        ascending=[True, True, False],
    )
    top_teammates = usage_ranked.groupby(["team_abbr", "game_date"], as_index=False).head(3)
    top_teammates = top_teammates.rename(columns={"player": "teammate"})[
        ["team_abbr", "game_date", "teammate"]
    ]

    injuries_out = injuries[injuries["is_out"] == 1].copy()
    injuries_out = injuries_out.rename(columns={"player": "teammate"})

    player_rows = out[["player", "team_abbr", "_game_date"]].drop_duplicates().rename(columns={"_game_date": "game_date"})
    player_teammates = player_rows.merge(top_teammates, on=["team_abbr", "game_date"], how="left")
    player_teammates = player_teammates[player_teammates["player"] != player_teammates["teammate"]]
    top_out_matches = player_teammates.merge(
        injuries_out,
        left_on=["team_abbr", "teammate", "game_date"],
        right_on=["team_abbr", "teammate", "report_date"],
        how="left",
    )
    top_out_flag = (
        top_out_matches.groupby(["player", "team_abbr", "game_date"])["status"]
        .apply(lambda s: float(s.notna().any()))
        .reset_index(name="top_teammate_out_flag")
    )
    out = out.merge(top_out_flag, on=["player", "team_abbr", "game_date"], how="left")
    out["top_teammate_out_flag"] = pd.to_numeric(out["top_teammate_out_flag"], errors="coerce").fillna(0.0)

    out_players = injuries_out.merge(
        usage,
        left_on=["team_abbr", "teammate", "report_date"],
        right_on=["team_abbr", "player", "game_date"],
        how="left",
    )
    out_players["min_r10"] = pd.to_numeric(out_players["min_r10"], errors="coerce").fillna(0.0)
    out_minutes = (
        out_players.groupby(["team_abbr", "report_date"], as_index=False)["min_r10"]
        .sum()
        .rename(columns={"min_r10": "out_teammates_min_proxy", "report_date": "game_date"})
    )

    out = out.merge(out_minutes, on=["team_abbr", "game_date"], how="left")
    out["out_teammates_min_proxy"] = pd.to_numeric(out["out_teammates_min_proxy"], errors="coerce").fillna(0.0)

    out = out.drop(columns=["_game_date"], errors="ignore")

    return out


def main():
    features_path = Path("artifacts") / "features_v1.parquet"
    injuries_path = Path("artifacts") / "injuries_latest.csv"
    if not features_path.exists():
        print("Missing artifacts/features_v1.parquet")
        return
    df = pd.read_parquet(features_path)
    df = add_availability_features(df, injuries_path)
    df.to_parquet(features_path, index=False)
    print("Saved availability features into", features_path)


if __name__ == "__main__":
    main()
