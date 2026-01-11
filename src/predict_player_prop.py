import json
import math
import re
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
    return 6.0 if stat == "pts" else (2.2 if stat == "reb" else 1.8)

def tier(p: float) -> str:
    if p >= 0.60: return "A"
    if p >= 0.57: return "B"
    if p >= 0.55: return "C"
    return "D"

def parse_line_and_extras(line_str: str):
    """
    Supports:
      "22.5"
      "22.5@-120 platform=novig"
      "29.5 @ +105 platform=prophetx"
      "6.5 platform=prizepicks"
    """
    s = line_str.strip().lower()

    # platform=...
    platform = "generic"
    m = re.search(r"platform\s*=\s*([a-z]+)", s)
    if m:
        platform = m.group(1).strip()

    # odds: find something that looks like -120 or +105 after @ or "odds="
    odds = None
    m = re.search(r"(?:@|odds\s*=)\s*([+-]?\d{2,5})", s)
    if m:
        try:
            odds = int(m.group(1))
        except Exception:
            odds = None

    # line: first number like 22.5
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        raise ValueError(f"Could not parse line from: {line_str}")
    line = float(m.group(1))

    return line, platform, odds

def implied_prob_from_american(odds: int) -> float:
    if odds == 0:
        return 0.0
    if odds < 0:
        a = abs(odds)
        return a / (a + 100.0)
    return 100.0 / (odds + 100.0)

def profit_per_1_from_american(odds: int) -> float:
    # profit on $1 stake
    if odds < 0:
        a = abs(odds)
        return 100.0 / a
    return odds / 100.0

def ev_per_1(p_win: float, odds: int) -> float:
    prof = profit_per_1_from_american(odds)
    return p_win * prof - (1.0 - p_win) * 1.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", required=True)
    parser.add_argument("--stat", required=True, choices=["pts", "reb", "ast"])
    parser.add_argument("--line", required=True)  # accept string so we can parse extras
    args = parser.parse_args()

    try:
        line, platform, odds = parse_line_and_extras(args.line)

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

        min_pred = float(np.clip(min_model.predict(X)[0], 0.0, 48.0))
        rate_pred = float(np.clip(rate_model.predict(X)[0], 0.0, 10.0))
        proj = min_pred * rate_pred

        rmse = load_rmse_v2(args.stat)
        z = (line - proj) / rmse if rmse > 1e-9 else 0.0
        p_over = 1.0 - normal_cdf(z)
        p_under = 1.0 - p_over

        best_side = "PASS"
        best_p = max(p_over, p_under)
        if best_p >= 0.55:
            best_side = "OVER" if p_over >= p_under else "UNDER"

        # Platform formatting
        odds_mode = platform in {"novig", "prophetx", "fliff"}
        pickem_mode = platform in {"underdog", "betr", "prizepicks"}

        chosen_p = p_over if best_side == "OVER" else (p_under if best_side == "UNDER" else 0.0)

        ev = None
        edge = None
        implied = None
        if odds_mode and odds is not None and best_side in {"OVER", "UNDER"}:
            implied = implied_prob_from_american(odds)
            edge = float(chosen_p - implied)
            ev = float(ev_per_1(chosen_p, odds))

        slip_hint = None
        if pickem_mode and best_side in {"OVER", "UNDER"}:
            # simple guidance (correlation layer comes later with pbpstats)
            t = tier(chosen_p)
            if t == "A":
                slip_hint = "Good anchor leg. Use in 2–3 leg slips; avoid pairing with same-game teammates until correlation layer is added."
            elif t == "B":
                slip_hint = "Solid 2-leg candidate. Prefer different games/teams."
            elif t == "C":
                slip_hint = "Only use if you need a filler; prefer 2-leg max."
            else:
                slip_hint = "Not strong enough for pick’em slips."

        report = {
            "model_version": "v2_minutes_x_rate + v2_context_features",
            "player_query": args.player,
            "matched_player": str(last["player"]),
            "stat": args.stat,
            "line": line,
            "platform": platform,
            "odds_american": odds,
            "projection": float(proj),
            "minutes_pred": float(min_pred),
            "rate_pred": float(rate_pred),
            "rmse_used": float(rmse),
            "p_over": float(p_over),
            "p_under": float(p_under),
            "recommendation": best_side,
            "confidence_tier": tier(best_p) if best_side != "PASS" else "PASS",
            "chosen_prob": float(chosen_p) if best_side != "PASS" else None,
            "implied_prob": implied,
            "edge_prob": edge,
            "ev_per_1": ev,
            "pickem_slip_hint": slip_hint,
            "last_game_date_used": str(pd.to_datetime(last["game_date"]).date()),
            "feature_notes": {
                "rest_days": float(last.get("rest_days", 0.0)),
                "b2b": int(last.get("b2b", 0)),
                "games_last_7d": float(last.get("games_last_7d", 0.0)),
                "team_drtg": float(last.get("team_drtg", 0.0)),
                "opp_drtg": float(last.get("opp_drtg", 0.0)),
            },
            "next_sharp_layers": [
                "official injury report ingest",
                "lineup/on-off & teammate-out boosts (pbpstats / nba-on-court)",
                "true EV backtest vs odds (Novig/ProphetX/Fliff)",
                "correlation-aware slip builder (Underdog/PrizePicks/Betr)"
            ]
        }

        ART_DIR.mkdir(parents=True, exist_ok=True)
        (ART_DIR / "player_pick.json").write_text(json.dumps(report, indent=2))
        (ART_DIR / "player_query_report.json").write_text(json.dumps(report, indent=2))

        # small human-readable summary
        summary = {
            "summary": f"{report['matched_player']} {report['stat'].upper()} {report['line']} -> {report['recommendation']} "
                       f"(P={report['chosen_prob']:.3f} tier={report['confidence_tier']}) proj={report['projection']:.2f} "
                       f"(min={report['minutes_pred']:.1f}, rate={report['rate_pred']:.3f})",
        }
        (ART_DIR / "player_query_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(report, indent=2))

    except Exception as e:
        write_error(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
