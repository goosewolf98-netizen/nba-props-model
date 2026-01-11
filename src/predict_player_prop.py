import json
import math
import argparse
from pathlib import Path

import numpy as np
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

def load_rmse_v2(stat: str) -> float:
    # Prefer walk-forward v2 rmse
    p = ART_DIR / "walkforward_metrics_v2.json"
    if p.exists():
        j = json.loads(p.read_text())
        try:
            return float(j[stat]["overall"]["rmse"])
        except Exception:
            pass

    # Fallback: holdout v2 rmse
    p = ART_DIR / "backtest_metrics_v2.json"
    if p.exists():
        j = json.loads(p.read_text())
        try:
            return float(j[stat]["rmse"])
        except Exception:
            pass

    # Last resort
    return 4.5 if stat == "pts" else (2.0 if stat == "reb" else 1.6)

def tier(p: float) -> str:
    if p >= 0.60: return "A"
    if p >= 0.57: return "B"
    if p >= 0.55: return "C"
    return "D"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", required=True)
    parser.add_argument("--stat", required=True, choices=["pts", "reb", "ast"])
    parser.add_argument("--line", required=True, type=float)
    parser.add_argument("--platform", required=False, default="generic",
                        choices=["generic","novig","prophetx","underdog","betr","prizepicks","fliff"])
    args = parser.parse_args()

    try:
        feat_path = ART_DIR / "features_v1.parquet"
        if not feat_path.exists():
            raise FileNotFoundError("Missing artifacts/features_v1.parquet. Run Feature Factory first.")

        df = pd.read_parquet(feat_path)
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        df = df.dropna(subset=["game_date"])

        last = find_last_row(df, args.player)

        cols_path = ART_DIR / "feature_cols_v2.json"
        min_model_path = ART_DIR / "xgb_min.joblib"
        rate_model_path = ART_DIR / f"xgb_{args.stat}_rate.joblib"

        if not (cols_path.exists() and min_model_path.exists() and rate_model_path.exists()):
            raise FileNotFoundError("Missing v2 artifacts (feature_cols_v2.json / xgb_min.joblib / xgb_*_rate.joblib). Run training step.")

        feature_cols = json.loads(cols_path.read_text())
        min_model = load(min_model_path)
        rate_model = load(rate_model_path)

        row = {c: float(last.get(c, 0.0)) for c in feature_cols}
        X = pd.DataFrame([row]).apply(pd.to_numeric, errors="coerce").fillna(0.0)

        min_pred = float(min_model.predict(X)[0])
        rate_pred = float(rate_model.predict(X)[0])

        min_pred = float(np.clip(min_pred, 0.0, 48.0))
        rate_pred = float(np.clip(rate_pred, 0.0, 10.0))

        proj = min_pred * rate_pred

        rmse = load_rmse_v2(args.stat)
        z = (args.line - proj) / rmse if rmse > 1e-9 else 0.0
        p_over = 1.0 - normal_cdf(z)
        p_under = 1.0 - p_over

        best_side = "PASS"
        best_p = max(p_over, p_under)
        if p_over >= 0.55:
            best_side = "OVER"
        if p_under >= 0.55 and p_under > p_over:
            best_side = "UNDER"
        if best_p < 0.55:
            best_side = "PASS"

        out = {
            "model_version": "v2_minutes_x_rate",
            "platform": args.platform,
            "player_query": args.player,
            "matched_player": str(last["player"]),
            "stat": args.stat,
            "line": args.line,
            "projection": proj,
            "minutes_pred": min_pred,
            "rate_pred": rate_pred,
            "rmse_used": rmse,
            "p_over": p_over,
            "p_under": p_under,
            "recommendation": best_side,
            "confidence_tier": tier(best_p) if best_side != "PASS" else "PASS",
            "last_game_date_used": str(pd.to_datetime(last["game_date"]).date()),
            "note": "V2 = minutes × per-minute rate. Next sharp layers: official injury report + lineup/on-off + EV vs odds."
        }

        (ART_DIR / "player_pick.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))

    except Exception as e:
        write_error(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
