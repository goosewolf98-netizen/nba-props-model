from __future__ import annotations

from pathlib import Path
import json
import math
import argparse
import pandas as pd


ART_DIR = Path("artifacts")


def _normalize_player(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def _normalize_market(market: str) -> str:
    market = str(market).strip().lower()
    mapping = {"points": "pts", "point": "pts", "pts": "pts",
               "rebounds": "reb", "rebound": "reb", "reb": "reb",
               "assists": "ast", "assist": "ast", "ast": "ast"}
    return mapping.get(market, market)


def _prob_over(proj: float, line: float, rmse: float) -> float:
    if rmse <= 1e-9:
        return 0.5
    z = (line - proj) / rmse
    return 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))


def _load_rmse(stat: str) -> float:
    metrics_path = ART_DIR / "walkforward_metrics_v2.json"
    if metrics_path.exists():
        try:
            data = json.loads(metrics_path.read_text())
            return float(data[stat]["overall"]["rmse"])
        except Exception:
            pass
    fallback = {"pts": 6.0, "reb": 2.2, "ast": 1.8}
    return fallback.get(stat, 3.0)


def run_calibration(pred_path: Path, lines_path: Path) -> dict:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    if not pred_path.exists():
        note = {"error": "missing_predictions", "pred_path": str(pred_path)}
        (ART_DIR / "calibration_report.json").write_text(json.dumps(note, indent=2))
        return note

    preds = pd.read_csv(pred_path)
    if preds.empty:
        note = {"error": "empty_predictions", "pred_path": str(pred_path)}
        (ART_DIR / "calibration_report.json").write_text(json.dumps(note, indent=2))
        pd.DataFrame().to_csv(ART_DIR / "reliability_curve.csv", index=False)
        return note

    for col in ["game_date", "player", "stat", "projection", "p_over", "p_under", "actual"]:
        if col not in preds.columns:
            preds[col] = pd.NA

    preds["game_date"] = pd.to_datetime(preds["game_date"], errors="coerce").dt.date.astype(str)
    preds["player_norm"] = preds["player"].apply(_normalize_player)
    preds["stat_norm"] = preds["stat"].apply(_normalize_market)
    preds["projection"] = pd.to_numeric(preds["projection"], errors="coerce")
    preds["p_over"] = pd.to_numeric(preds["p_over"], errors="coerce")
    preds["p_under"] = pd.to_numeric(preds["p_under"], errors="coerce")
    preds["actual"] = pd.to_numeric(preds["actual"], errors="coerce")

    if not lines_path.exists():
        note = {"note": "lines missing - skipped", "lines_path": str(lines_path)}
        (ART_DIR / "calibration_report.json").write_text(json.dumps(note, indent=2))
        pd.DataFrame().to_csv(ART_DIR / "reliability_curve.csv", index=False)
        return note

    try:
        lines = pd.read_csv(lines_path)
    except Exception:
        lines = pd.DataFrame()
    if lines.empty:
        note = {"note": "lines missing - skipped", "lines_path": str(lines_path)}
        (ART_DIR / "calibration_report.json").write_text(json.dumps(note, indent=2))
        pd.DataFrame().to_csv(ART_DIR / "reliability_curve.csv", index=False)
        return note
    for col in ["game_date", "player", "stat", "line"]:
        if col not in lines.columns:
            lines[col] = pd.NA
    lines["game_date"] = pd.to_datetime(lines["game_date"], errors="coerce").dt.date.astype(str)
    lines["player"] = lines["player"].astype(str)
    lines["player_norm"] = lines["player"].apply(_normalize_player)
    lines["stat"] = lines["stat"].astype(str)
    lines["stat_norm"] = lines["stat"].apply(_normalize_market)
    lines["line"] = pd.to_numeric(lines["line"], errors="coerce")

    merged = preds.merge(
        lines,
        left_on=["game_date", "player_norm", "stat_norm"],
        right_on=["game_date", "player_norm", "stat_norm"],
        how="left",
    )
    merged["line"] = pd.to_numeric(merged["line"], errors="coerce")

    valid = merged.dropna(subset=["line", "projection", "actual"])
    if valid.empty:
        note = {"error": "no_matching_lines", "lines_path": str(lines_path)}
        (ART_DIR / "calibration_report.json").write_text(json.dumps(note, indent=2))
        pd.DataFrame().to_csv(ART_DIR / "reliability_curve.csv", index=False)
        return note

    def compute_p_over(row):
        if not pd.isna(row["p_over"]):
            return row["p_over"]
        rmse = _load_rmse(row["stat_norm"])
        return _prob_over(row["projection"], row["line"], rmse)

    valid["p_over"] = valid.apply(compute_p_over, axis=1)
    valid["outcome"] = (valid["actual"] > valid["line"]).astype(int)

    brier = float(((valid["p_over"] - valid["outcome"]) ** 2).mean())
    valid["p_over_clipped"] = valid["p_over"].clip(1e-6, 1 - 1e-6)
    logloss = float(
        -(valid["outcome"] * valid["p_over_clipped"].apply(math.log)
          + (1 - valid["outcome"]) * (1 - valid["p_over_clipped"]).apply(math.log)).mean()
    )

    bins = pd.cut(
        valid["p_over"],
        bins=[0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
        include_lowest=True,
    )
    curve = valid.groupby(bins).agg(
        avg_pred=("p_over", "mean"),
        empirical_win=("outcome", "mean"),
        count=("outcome", "size"),
    ).reset_index().rename(columns={"p_over": "bin"})
    curve.to_csv(ART_DIR / "reliability_curve.csv", index=False)

    report = {
        "brier": brier,
        "logloss": logloss,
        "rows": int(len(valid)),
    }
    (ART_DIR / "calibration_report.json").write_text(json.dumps(report, indent=2))
    print("Saved calibration outputs:", ART_DIR / "calibration_report.json")
    return report


def main():
    parser = argparse.ArgumentParser(description="Calibration report based on walk-forward predictions.")
    parser.add_argument("--predictions", default=str(ART_DIR / "walkforward_predictions_v2.csv"))
    parser.add_argument("--lines", default="data/lines/sdi_props_closing.csv")
    args = parser.parse_args()
    run_calibration(Path(args.predictions), Path(args.lines))


if __name__ == "__main__":
    main()
