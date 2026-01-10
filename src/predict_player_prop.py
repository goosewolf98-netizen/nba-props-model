import json
import math
import argparse
from pathlib import Path

import pandas as pd
from joblib import load

RAW_DIR = Path("data/raw")
ART_DIR = Path("artifacts")

def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def write_error(msg: str):
    ART_DIR.mkdir(parents=True, exist_ok=True)
    (ART_DIR / "player_pick_error.txt").write_text(msg)
    print(msg)

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    for c in cols:
        cl = c.lower()
        for cand in candidates:
            if cand in cl:
                return c
    return None

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    player_c = pick_col(df, ["player", "player_name", "athlete", "athlete_name", "athlete_display_name", "name"])
    date_c   = pick_col(df, ["game_date", "date", "start_date", "game_datetime"])
    min_c    = pick_col(df, ["min", "minutes", "mp"])
    pts_c    = pick_col(df, ["pts", "points"])
    reb_c    = pick_col(df, ["reb", "rebounds", "trb", "rebs", "total_rebounds"])
    ast_c    = pick_col(df, ["ast", "assists"])

    missing = [k for k,v in {
        "player": player_c, "game_date": date_c, "min": min_c,
        "pts": pts_c, "reb": reb_c, "ast": ast_c
    }.items() if v is None]
    if missing:
        raise ValueError(f"Missing columns {missing}. Have: {list(df.columns)[:50]}")

    df = df.rename(columns={
        player_c: "player",
        date_c: "game_date",
        min_c: "min",
        pts_c: "pts",
        reb_c: "reb",
        ast_c: "ast",
    }).copy()

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["game_date"])
    for c in ["min", "pts", "reb", "ast"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player", "game_date"]).copy()
    grp = df.groupby("player", group_keys=False)

    for stat in ["min", "pts", "reb", "ast"]:
        df[f"{stat}_r5"]   = grp[stat].shift(1).rolling(5,  min_periods=1).mean()
        df[f"{stat}_r10"]  = grp[stat].shift(1).rolling(10, min_periods=1).mean()
        df[f"{stat}_sd10"] = grp[stat].shift(1).rolling(10, min_periods=2).std()

    df["gp_last14"] = grp["game_date"].shift(1).rolling(14, min_periods=1).count()
    return df.fillna(0)

def find_data_file() -> Path:
    files = sorted(RAW_DIR.glob("*.csv")) + sorted(RAW_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError("No data found in data/raw.")
    return files[0]

def find_player_last_game(df: pd.DataFrame, player_query: str) -> pd.Series:
    pq = player_query.strip().lower()
    names = df["player"].astype(str)

    exact = df[names.str.lower() == pq]
    if len(exact) > 0:
        return exact.sort_values("game_date").iloc[-1]

    contains = df[names.str.lower().str.contains(pq, na=False)]
    if len(contains) > 0:
        return contains.sort_values("game_date").iloc[-1]

    sample = sorted(df["player"].dropna().astype(str).unique().tolist())
    raise ValueError(f"Player not found: '{player_query}'. Examples: {sample[:25]}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", required=True)
    parser.add_argument("--stat", required=True, choices=["pts", "reb", "ast"])
    parser.add_argument("--line", required=True, type=float)
    args = parser.parse_args()

    try:
        ART_DIR.mkdir(parents=True, exist_ok=True)

        data_path = find_data_file()
        df = pd.read_csv(data_path) if data_path.suffix == ".csv" else pd.read_parquet(data_path)
        df = standardize_columns(df)
        df = add_rolling_features(df)

        last = find_player_last_game(df, args.player)

        model_path = ART_DIR / f"xgb_{args.stat}.joblib"
        metrics_path = ART_DIR / "backtest_metrics.json"
        if not model_path.exists() or not metrics_path.exists():
            raise FileNotFoundError("Missing trained models/metrics. Run training first.")

        model = load(model_path)
        metrics = json.loads(metrics_path.read_text())
        rmse = float(metrics[args.stat]["rmse"]) if args.stat in metrics else 5.0

        feature_cols = [c for c in df.columns if c.endswith(("_r5", "_r10", "_sd10"))] + ["gp_last14"]

        # IMPORTANT FIX: build a numeric dataframe (avoids object dtype)
        X = pd.DataFrame([{c: last[c] for c in feature_cols}])
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        proj = float(model.predict(X)[0])

        z = (args.line - proj) / rmse if rmse > 1e-9 else 0.0
        p_over = 1.0 - normal_cdf(z)
        p_under = 1.0 - p_over

        out = {
            "player_query": args.player,
            "matched_player": str(last["player"]),
            "stat": args.stat,
            "line": args.line,
            "projection": proj,
            "rmse_used": rmse,
            "p_over": p_over,
            "p_under": p_under,
            "last_game_date_used": str(last["game_date"].date()),
            "note": "Baseline only. Next upgrades add injuries/minutes/teammate/matchup/EV."
        }

        (ART_DIR / "player_pick.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))

    except Exception as e:
        write_error(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
