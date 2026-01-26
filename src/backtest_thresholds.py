from __future__ import annotations

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

ART_DIR = Path("artifacts")

THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5]


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _synthetic_line(df: pd.DataFrame) -> pd.Series:
    # Synthetic line: projection rounded to nearest 0.5
    return np.round(df["y_pred"] * 2.0) / 2.0


def _tier_from_prob(prob: pd.Series) -> pd.Series:
    tiers = pd.Series(["C"] * len(prob), index=prob.index)
    tiers[prob >= 0.62] = "A"
    tiers[(prob >= 0.57) & (prob < 0.62)] = "B"
    return tiers


def run_backtest(
    pred_path: Path,
    art_dir: Path = ART_DIR,
) -> dict:
    art_dir.mkdir(parents=True, exist_ok=True)
    if not pred_path.exists():
        print(f"Missing predictions file: {pred_path}")
        return {"error": "missing_predictions", "pred_path": str(pred_path)}

    df = pd.read_csv(pred_path)
    if "stat" not in df.columns:
        df["stat"] = "unknown"
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    if "y_pred" not in df.columns:
        if {"min_pred", "rate_pred"}.issubset(df.columns):
            df["y_pred"] = _safe_numeric(df["min_pred"]) * _safe_numeric(df["rate_pred"])
        else:
            df["y_pred"] = 0.0
    df["y_pred"] = _safe_numeric(df["y_pred"])
    if "y_true" not in df.columns:
        df["y_true"] = 0.0
    df["y_true"] = _safe_numeric(df["y_true"])

    if "chosen_prob" in df.columns:
        df["chosen_prob"] = _safe_numeric(df["chosen_prob"], default=np.nan)
        df["tier"] = _tier_from_prob(df["chosen_prob"])
    else:
        df["chosen_prob"] = np.nan
        df["tier"] = "All"

    df["line"] = _synthetic_line(df)
    df["margin"] = df["y_pred"] - df["line"]

    results = []
    rows = []

    for stat in sorted(df["stat"].unique()):
        sub = df[df["stat"] == stat]
        if len(sub) == 0:
            continue
        for t in THRESHOLDS:
            for side, mask in {
                "over": sub["margin"] >= t,
                "under": sub["margin"] <= -t,
            }.items():
                bet_df = sub[mask].copy()
                if len(bet_df) == 0:
                    continue
                if side == "over":
                    bet_df["win"] = (bet_df["y_true"] > bet_df["line"]).astype(int)
                else:
                    bet_df["win"] = (bet_df["y_true"] < bet_df["line"]).astype(int)

                for tier in sorted(bet_df["tier"].unique()):
                    tier_df = bet_df[bet_df["tier"] == tier]
                    if len(tier_df) == 0:
                        continue
                    win_pct = float(tier_df["win"].mean()) if len(tier_df) else 0.0
                    avg_margin = float(tier_df["margin"].mean()) if len(tier_df) else 0.0
                    record = {
                        "stat": stat,
                        "threshold": t,
                        "side": side,
                        "tier": tier,
                        "bets": int(len(tier_df)),
                        "win_pct": win_pct,
                        "avg_margin": avg_margin,
                        "line_mode": "projection_round",
                    }
                    results.append(record)
                    rows.append(record)

    summary = {
        "settings": {
            "line_mode": "projection_round",
            "thresholds": THRESHOLDS,
        },
        "results": results,
    }

    summary_path = art_dir / "threshold_backtest_summary.json"
    detail_path = art_dir / "threshold_backtest_by_margin.csv"
    summary_path.write_text(json.dumps(summary, indent=2))
    pd.DataFrame(rows).to_csv(detail_path, index=False)

    print(f"Saved threshold backtest summary: {summary_path}")
    print(f"Saved threshold backtest detail: {detail_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Threshold backtest on walk-forward projections.")
    parser.add_argument(
        "--predictions",
        default=str(ART_DIR / "walkforward_predictions_v2.csv"),
        help="Path to walkforward_predictions_v2.csv",
    )
    args = parser.parse_args()

    run_backtest(Path(args.predictions), ART_DIR)


if __name__ == "__main__":
    main()
