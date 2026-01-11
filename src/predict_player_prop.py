import json
import math
import argparse
from pathlib import Path

import pandas as pd
from joblib import load

ART_DIR = Path("artifacts")

def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def write_error(msg: str):
    ART_DIR.mkdir(parents=True, exist_ok=True)
    (ART_DIR / "player_pick_error.txt").write_text(msg)
    print(msg)

def find_last_row(df: pd.DataFrame, player_query: str) -> pd.Series:
    pq = player_query.strip().lower()
    names = df["player"].astype(str)
    exact = df[names.str.lower() == pq]
    if len(exact) > 0:
        return exact.sort_values("game_date").iloc[-1]
    contains = df[names.str.lower().str.contains(pq, na=False)]
    if len(contains) > 0:
        return contains.sort_values("game_date").iloc[-1]
    sample = sorted(df["player"].dropna().astype(str).unique().tolist())[:25]
    raise ValueError(f"Player not found: '{player_query}'. Examples: {sample}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", required=True)
    parser.add_argument("--stat", required=True, choices=["pts", "reb", "ast"])
    parser.add_argument("--line", required=True, type=float)
    args = parser.parse_args()

    try:
        feat_path = ART_DIR / "features_v1.parquet"
        if not feat_path.exists():
            raise FileNotFoundError("Missing artifacts/features_v1.parquet. Run Feature Factory first.")

        df = pd.read_parquet(feat_path)
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        df = df.dropna(subset=["game_date"])

        last = find_last_row(df, args.player)

        model_path = ART_DIR / f"xgb_{args.stat}.joblib"
        metrics_path = ART_DIR / "backtest_metrics.json"
        cols_path = ART_DIR / "feature_cols.json"

        if not model_path.exists() or not metrics_path.exists() or not cols_path.exists():
            raise FileNotFoundError("Missing model/metrics/feature_cols. Training must run first.")

        model = load(model_path)
        metrics = json.loads(metrics_path.read_text())
        feature_cols = json.loads(cols_path.read_text())

        rmse = float(metrics[args.stat]["rmse"])

        row = {c: last.get(c, 0.0) for c in feature_cols}
        X = pd.DataFrame([row]).apply(pd.to_numeric, errors="coerce").fillna(0.0)

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
            "note": "v1 features: includes team context + stability. Next: injuries/minutes/teammate-on-off/EV."
        }

        (ART_DIR / "player_pick.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))

    except Exception as e:
        write_error(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
