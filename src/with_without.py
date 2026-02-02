from __future__ import annotations

from pathlib import Path
import pandas as pd

from schema_normalize import norm_all, norm_minutes


RAW_PLAYER = Path("data/raw/nba_player_box.csv")
AVAIL_PATH = Path("data/injuries/availability_by_game.csv")
OUT_PATH = Path("artifacts/with_without_features.parquet")


def _load_player_box() -> pd.DataFrame:
    path = RAW_PLAYER
    if not path.exists():
        path = RAW_PLAYER.parent / "nba_player_box_2024_25_and_2025_26.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
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
    if "team_abbr" not in pb.columns:
        pb = norm_all(pb)
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
    top_usage = team_usage.sort_values(["team_abbr", "pts"], ascending=[True, False]).groupby("team_abbr").head(3)
    top_ast = team_usage.sort_values(["team_abbr", "ast"], ascending=[True, False]).groupby("team_abbr").head(3)
    top_reb = team_usage.sort_values(["team_abbr", "reb"], ascending=[True, False]).groupby("team_abbr").head(3)

    avail_out = avail[avail["is_out"] == 1].copy()

    # 1. Starters/Rotation out count
    out_counts = avail_out.groupby(["team_abbr", "game_date"]).size().reset_index(name="out_count")

    # 2. Top Usage/Ast/Reb Out helper
    def get_top_out_counts(top_df, col_name_base):
        top_out = avail_out.merge(top_df[["team_abbr", "player"]], on=["team_abbr", "player"], how="inner")
        top_out_counts = top_out.groupby(["team_abbr", "game_date"]).size().reset_index(name=f"{col_name_base}_base")
        top_out["is_in_top"] = 1
        return top_out_counts, top_out[["team_abbr", "game_date", "player", "is_in_top"]]

    usage_counts, usage_details = get_top_out_counts(top_usage, "top_usage_out")
    ast_counts, ast_details = get_top_out_counts(top_ast, "top_ast_out")
    reb_counts, reb_details = get_top_out_counts(top_reb, "top_reb_out")

    # Merge everything into pb
    out_df = pb.copy()

    out_df = out_df.merge(out_counts, on=["team_abbr", "game_date"], how="left")
    out_df["starters_out_count"] = out_df["out_count"].fillna(0).astype(int)
    out_df["rotation_out_count"] = out_df["starters_out_count"]

    # Merge top usage
    out_df = out_df.merge(usage_counts, on=["team_abbr", "game_date"], how="left")
    out_df = out_df.merge(usage_details.rename(columns={"is_in_top": "is_usage_out"}),
                  on=["team_abbr", "game_date", "player"], how="left")
    out_df["top_usage_teammates_out_count"] = (out_df["top_usage_out_base"].fillna(0) - out_df["is_usage_out"].fillna(0)).astype(int)

    # Merge top ast
    out_df = out_df.merge(ast_counts, on=["team_abbr", "game_date"], how="left")
    out_df = out_df.merge(ast_details.rename(columns={"is_in_top": "is_ast_out"}),
                  on=["team_abbr", "game_date", "player"], how="left")
    out_df["top_ast_teammates_out_count"] = (out_df["top_ast_out_base"].fillna(0) - out_df["is_ast_out"].fillna(0)).astype(int)

    # Merge top reb
    out_df = out_df.merge(reb_counts, on=["team_abbr", "game_date"], how="left")
    out_df = out_df.merge(reb_details.rename(columns={"is_in_top": "is_reb_out"}),
                  on=["team_abbr", "game_date", "player"], how="left")
    out_df["top_reb_teammates_out_count"] = (out_df["top_reb_out_base"].fillna(0) - out_df["is_reb_out"].fillna(0)).astype(int)

    final_cols = [
        "game_date", "team_abbr", "player",
        "starters_out_count", "rotation_out_count",
        "top_usage_teammates_out_count",
        "top_ast_teammates_out_count",
        "top_reb_teammates_out_count",
    ]
    out_df = out_df[final_cols]
    out_df = out_df.drop_duplicates(subset=["game_date", "team_abbr", "player"])
    out_df.to_parquet(OUT_PATH, index=False)
    print("Saved with/without features to", OUT_PATH)


if __name__ == "__main__":
    main()
