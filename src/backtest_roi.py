from __future__ import annotations

from pathlib import Path
import json
import math
import argparse
import pandas as pd


ART_DIR = Path("artifacts")
DEFAULT_ODDS = -110
THRESHOLDS = [0.01, 0.02, 0.03, 0.04, 0.05]


def _normalize_player(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def _normalize_market(market: str) -> str:
    market = str(market).strip().lower()
    mapping = {"points": "pts", "point": "pts", "pts": "pts",
               "rebounds": "reb", "rebound": "reb", "reb": "reb",
               "assists": "ast", "assist": "ast", "ast": "ast"}
    return mapping.get(market, market)


def _implied_prob(odds: float) -> float:
    if odds is None or math.isnan(odds):
        odds = DEFAULT_ODDS
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 100.0 / (odds + 100.0)


def _payout(odds: float) -> float:
    if odds is None or math.isnan(odds):
        odds = DEFAULT_ODDS
    if odds < 0:
        return 100.0 / abs(odds)
    return odds / 100.0


def _tier(prob: float) -> str:
    if prob >= 0.60:
        return "A"
    if prob >= 0.57:
        return "B"
    if prob >= 0.55:
        return "C"
    return "D"


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


def _prob_over(proj: float, line: float, rmse: float) -> float:
    if rmse <= 1e-9:
        return 0.5
    z = (line - proj) / rmse
    return 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))


def run_backtest(pred_path: Path, lines_path: Path) -> dict:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    def write_skipped(reason: str, detail: str | None = None) -> dict:
        note = {"status": "skipped", "reason": reason}
        if detail:
            note["detail"] = detail
        (ART_DIR / "roi_backtest_summary.json").write_text(json.dumps(note, indent=2))
        pd.DataFrame().to_csv(ART_DIR / "roi_backtest_by_edge.csv", index=False)
        pd.DataFrame().to_csv(ART_DIR / "roi_backtest_bets.csv", index=False)
        return note

    if not pred_path.exists():
        note = {"error": "missing_predictions", "pred_path": str(pred_path)}
        (ART_DIR / "roi_backtest_summary.json").write_text(json.dumps(note, indent=2))
        return note

    preds = pd.read_csv(pred_path)
    if preds.empty:
        note = {"error": "empty_predictions", "pred_path": str(pred_path)}
        (ART_DIR / "roi_backtest_summary.json").write_text(json.dumps(note, indent=2))
        pd.DataFrame().to_csv(ART_DIR / "roi_backtest_by_edge.csv", index=False)
        pd.DataFrame().to_csv(ART_DIR / "roi_backtest_bets.csv", index=False)
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
        return write_skipped("no lines matched", f"missing lines file: {lines_path}")

    try:
        lines = pd.read_csv(lines_path)
    except Exception:
        lines = pd.DataFrame()
    if lines.empty:
        return write_skipped("no lines matched", "lines file empty")
    for col in ["game_date", "player", "stat", "line", "over_odds", "under_odds"]:
        if col not in lines.columns:
            lines[col] = pd.NA
    lines["game_date"] = pd.to_datetime(lines["game_date"], errors="coerce").dt.date.astype(str)
    lines["player"] = lines["player"].astype(str)
    lines["player_norm"] = lines["player"].apply(_normalize_player)
    lines["stat"] = lines["stat"].astype(str)
    lines["stat_norm"] = lines["stat"].apply(_normalize_market)
    lines["line"] = pd.to_numeric(lines["line"], errors="coerce")
    lines["over_odds"] = pd.to_numeric(lines["over_odds"], errors="coerce").fillna(DEFAULT_ODDS)
    lines["under_odds"] = pd.to_numeric(lines["under_odds"], errors="coerce").fillna(DEFAULT_ODDS)

    merged = preds.merge(
        lines,
        left_on=["game_date", "player_norm", "stat_norm"],
        right_on=["game_date", "player_norm", "stat_norm"],
        how="left",
    )
    merged["line"] = pd.to_numeric(merged["line"], errors="coerce")
    merged["over_odds"] = pd.to_numeric(merged["over_odds"], errors="coerce").fillna(DEFAULT_ODDS)
    merged["under_odds"] = pd.to_numeric(merged["under_odds"], errors="coerce").fillna(DEFAULT_ODDS)

    valid = merged.dropna(subset=["line", "projection"])
    if valid.empty:
        return write_skipped("no lines matched", "no matching prediction/line rows")

    def compute_probs(row):
        if not pd.isna(row["p_over"]) and not pd.isna(row["p_under"]):
            return row["p_over"], row["p_under"]
        rmse = _load_rmse(row["stat_norm"])
        p_over = _prob_over(row["projection"], row["line"], rmse)
        return p_over, 1.0 - p_over

    probs = valid.apply(lambda r: compute_probs(r), axis=1, result_type="expand")
    valid["p_over"] = probs[0]
    valid["p_under"] = probs[1]

    valid["implied_over"] = valid["over_odds"].apply(_implied_prob)
    valid["implied_under"] = valid["under_odds"].apply(_implied_prob)
    valid["edge_over"] = valid["p_over"] - valid["implied_over"]
    valid["edge_under"] = valid["p_under"] - valid["implied_under"]

    bets = []
    summary_rows = []
    for threshold in THRESHOLDS:
        for side in ["over", "under"]:
            edge_col = "edge_over" if side == "over" else "edge_under"
            odds_col = "over_odds" if side == "over" else "under_odds"
            implied_col = "implied_over" if side == "over" else "implied_under"
            bet_df = valid[valid[edge_col] >= threshold].copy()
            if bet_df.empty:
                continue
            if "actual" not in bet_df.columns:
                continue
            bet_df = bet_df.dropna(subset=["actual", "line"])
            if bet_df.empty:
                continue
            if side == "over":
                bet_df["win"] = (bet_df["actual"] > bet_df["line"]).astype(int)
            else:
                bet_df["win"] = (bet_df["actual"] < bet_df["line"]).astype(int)
            bet_df["push"] = (bet_df["actual"] == bet_df["line"]).astype(int)
            bet_df["payout"] = bet_df[odds_col].apply(_payout)
            bet_df["profit"] = bet_df["win"] * bet_df["payout"] - (1 - bet_df["win"] - bet_df["push"])

            bet_df["tier"] = bet_df[["p_over", "p_under"]].max(axis=1).apply(_tier)
            bet_df["side"] = side
            bet_df["threshold"] = threshold
            bet_df["avg_edge"] = bet_df[edge_col]
            bet_df["break_even"] = bet_df[implied_col]
            bets.append(bet_df)

            for tier in sorted(bet_df["tier"].unique()):
                tier_df = bet_df[bet_df["tier"] == tier]
                roi = float(tier_df["profit"].sum() / len(tier_df)) if len(tier_df) else 0.0
                summary_rows.append({
                    "threshold": threshold,
                    "side": side,
                    "tier": tier,
                    "stat": "all",
                    "bets": int(len(tier_df)),
                    "win_pct": float(tier_df["win"].mean()) if len(tier_df) else 0.0,
                    "push_pct": float(tier_df["push"].mean()) if len(tier_df) else 0.0,
                    "roi": roi,
                    "avg_edge": float(tier_df[edge_col].mean()) if len(tier_df) else 0.0,
                    "break_even": float(tier_df[implied_col].mean()) if len(tier_df) else 0.0,
                })

            for stat in sorted(bet_df["stat_norm"].dropna().unique()):
                stat_df = bet_df[bet_df["stat_norm"] == stat]
                roi = float(stat_df["profit"].sum() / len(stat_df)) if len(stat_df) else 0.0
                summary_rows.append({
                    "threshold": threshold,
                    "side": side,
                    "tier": "all",
                    "stat": stat,
                    "bets": int(len(stat_df)),
                    "win_pct": float(stat_df["win"].mean()) if len(stat_df) else 0.0,
                    "push_pct": float(stat_df["push"].mean()) if len(stat_df) else 0.0,
                    "roi": roi,
                    "avg_edge": float(stat_df[edge_col].mean()) if len(stat_df) else 0.0,
                    "break_even": float(stat_df[implied_col].mean()) if len(stat_df) else 0.0,
                })

    bets_df = pd.concat(bets, ignore_index=True) if bets else pd.DataFrame()
    by_edge_df = pd.DataFrame(summary_rows)

    edge_bins = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 1.0]
    if not bets_df.empty:
        bets_df["edge_bucket"] = pd.cut(bets_df["avg_edge"], bins=edge_bins, include_lowest=True)
        edge_table = bets_df.groupby(["edge_bucket", "side", "stat_norm"]).agg(
            bets=("profit", "size"),
            win_pct=("win", "mean"),
            push_pct=("push", "mean"),
            roi=("profit", lambda x: float(x.sum() / len(x)) if len(x) else 0.0),
            avg_edge=("avg_edge", "mean"),
            break_even=("break_even", "mean"),
        ).reset_index()
    else:
        edge_table = pd.DataFrame()

    clv_path = Path("data/lines/props_line_movement.csv")
    clv_report = pd.DataFrame()
    clv_summary = {}
    avg_abs_line_move = 0.0
    if not bets_df.empty and clv_path.exists():
        try:
            moves = pd.read_csv(clv_path)
        except Exception:
            moves = pd.DataFrame()
        if not moves.empty:
            for col in ["game_date", "player", "stat", "book", "open_line", "close_line"]:
                if col not in moves.columns:
                    moves[col] = pd.NA
            moves["game_date"] = pd.to_datetime(moves["game_date"], errors="coerce").dt.date.astype(str)
            moves["player_norm"] = moves["player"].astype(str).apply(_normalize_player)
            moves["stat_norm"] = moves["stat"].astype(str).apply(_normalize_market)
            moves["open_line"] = pd.to_numeric(moves["open_line"], errors="coerce")
            moves["close_line"] = pd.to_numeric(moves["close_line"], errors="coerce")
            clv_report = bets_df.merge(
                moves[["game_date", "player_norm", "stat_norm", "book", "open_line", "close_line"]],
                on=["game_date", "player_norm", "stat_norm", "book"],
                how="left",
            )
            clv_report = clv_report.rename(columns={"open_line": "bet_line", "close_line": "close_line"})
            clv_report["bet_line"] = clv_report["bet_line"].fillna(clv_report["close_line"])
            clv_report["clv_line"] = clv_report.apply(
                lambda r: (r["close_line"] - r["bet_line"]) if r.get("side") == "over" else (r["bet_line"] - r["close_line"]),
                axis=1,
            )
            clv_report["line_move"] = clv_report["close_line"] - clv_report["bet_line"]
            avg_abs_line_move = float((moves["close_line"] - moves["open_line"]).abs().mean())
            clv_values = clv_report["clv_line"].dropna()
            clv_summary = {
                "avg_clv": float(clv_values.mean()) if len(clv_values) else 0.0,
                "median_clv": float(clv_values.median()) if len(clv_values) else 0.0,
                "pct_positive_clv": float((clv_values > 0).mean()) if len(clv_values) else 0.0,
            }
            if len(clv_values):
                clv_report["clv_bucket"] = pd.cut(
                    clv_report["clv_line"],
                    bins=[-float("inf"), -1.0, -0.5, 0.0, 0.5, 1.0, float("inf")],
                    include_lowest=True,
                )
                clv_bucket = clv_report.groupby("clv_bucket").agg(
                    bets=("profit", "size"),
                    roi=("profit", lambda x: float(x.sum() / len(x)) if len(x) else 0.0),
                ).reset_index()
                clv_summary["roi_by_clv_bucket"] = clv_bucket.to_dict(orient="records")

    overall_roi = float(bets_df["profit"].sum() / len(bets_df)) if len(bets_df) else 0.0
    summary = {
        "settings": {
            "thresholds": THRESHOLDS,
            "default_odds": DEFAULT_ODDS,
        },
        "overall": {
            "bets": int(len(bets_df)),
            "roi": overall_roi,
            "win_pct": float(bets_df["win"].mean()) if len(bets_df) else 0.0,
            "push_pct": float(bets_df["push"].mean()) if len(bets_df) else 0.0,
            "avg_edge": float(bets_df["avg_edge"].mean()) if len(bets_df) else 0.0,
            "break_even": float(bets_df["break_even"].mean()) if len(bets_df) else 0.0,
        },
        "by_threshold": summary_rows,
    }

    (ART_DIR / "roi_backtest_summary.json").write_text(json.dumps(summary, indent=2))
    edge_table.to_csv(ART_DIR / "roi_backtest_by_edge.csv", index=False)
    bets_df.to_csv(ART_DIR / "roi_backtest_bets.csv", index=False)
    clv_report.to_csv(ART_DIR / "clv_bets.csv", index=False)
    clv_bucket_df = pd.DataFrame(clv_summary.get("roi_by_clv_bucket", []))
    clv_bucket_df.to_csv(ART_DIR / "roi_by_clv_bucket.csv", index=False)
    (ART_DIR / "clv_summary.json").write_text(json.dumps(clv_summary, indent=2))

    market_diag = {
        "coverage": float(valid["line"].notna().mean()) if len(valid) else 0.0,
        "avg_abs_line_move": avg_abs_line_move,
        "edge_clv_correlation": float(clv_report[["avg_edge", "clv_line"]].corr().iloc[0, 1]) if len(clv_report) > 1 else 0.0,
    }
    if not clv_report.empty:
        clv_report["line_move_dir"] = clv_report["line_move"].apply(
            lambda x: "up" if pd.notna(x) and x > 0 else ("down" if pd.notna(x) and x < 0 else "flat")
        )
        market_diag["roi_by_line_move_dir"] = (
            clv_report.groupby(["side", "line_move_dir"]).agg(
                bets=("profit", "size"),
                roi=("profit", lambda x: float(x.sum() / len(x)) if len(x) else 0.0),
            ).reset_index().to_dict(orient="records")
        )
    if "book" in clv_report.columns and not clv_report.empty:
        roi_by_book = clv_report.groupby("book").agg(
            bets=("profit", "size"),
            roi=("profit", lambda x: float(x.sum() / len(x)) if len(x) else 0.0),
        ).reset_index()
        market_diag["roi_by_book"] = roi_by_book.to_dict(orient="records")
    (ART_DIR / "market_diagnostics.json").write_text(json.dumps(market_diag, indent=2))
    print("Saved ROI backtest outputs:", ART_DIR / "roi_backtest_summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description="ROI backtest using historical lines.")
    parser.add_argument("--predictions", default=str(ART_DIR / "walkforward_predictions_v2.csv"))
    parser.add_argument("--lines", default="data/lines/sdi_props_closing.csv")
    args = parser.parse_args()
    run_backtest(Path(args.predictions), Path(args.lines))


if __name__ == "__main__":
    main()
