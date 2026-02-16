import json
import math
import argparse
import re
from functools import lru_cache
import subprocess
import sys
import importlib.util
from pathlib import Path
from urllib.request import urlopen, Request

import numpy as np
import pandas as pd
from joblib import load
from scipy.stats import poisson

ART_DIR = Path("artifacts")
MODELS_DIR = Path("data/models")

@lru_cache(maxsize=None)
def get_model(path: Path):
    return load(path)

@lru_cache(maxsize=None)
def get_feature_cols(path: Path):
    return json.loads(path.read_text())

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
    p = ART_DIR / "walkforward_metrics_v2.json"
    if p.exists():
        j = json.loads(p.read_text())
        try:
            return float(j[stat]["overall"]["rmse"])
        except Exception:
            pass
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

# ---------------- Injury overlay ----------------

def http_get_text(url: str, timeout=20) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="ignore")

def http_get_bytes(url: str, timeout=30) -> bytes:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as r:
        return r.read()

def find_latest_injury_pdf():
    season_page = "https://official.nba.com/nba-injury-report-2025-26-season/"
    try:
        html = http_get_text(season_page)
        urls = re.findall(r"https?://[^\"'\s]+Injury-Report_\d{4}-\d{2}-\d{2}_[^\"'\s]+\.pdf", html)
        if not urls:
            rels = re.findall(r"/referee/injury/Injury-Report_\d{4}-\d{2}-\d{2}_[^\"'\s]+\.pdf", html)
            urls = ["https://ak-static.cms.nba.com" + r for r in rels]

        def key(u):
            m = re.search(r"Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})(AM|PM)\.pdf", u)
            if not m:
                return ("0000-00-00", 0, 0)
            d, hh, mm, ap = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
            h24 = hh % 12 + (12 if ap == "PM" else 0)
            return (d, h24, mm)

        urls = sorted(set(urls), key=key)
        return urls[-1] if urls else None
    except Exception:
        return None

def ensure_pypdf():
    if importlib.util.find_spec("pypdf") is not None:
        return True, None
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pypdf"])
    except Exception as e:
        return False, str(e)
    if importlib.util.find_spec("pypdf") is None:
        return False, "pypdf installation failed"
    return True, None

def parse_injury_status_from_text(text: str, player_full: str):
    player_full = str(player_full).strip()
    parts = player_full.split()
    first = parts[0] if parts else ""
    last = parts[-1] if parts else ""
    patterns = [
        f"{last}, {first}",
        player_full,
        last
    ]
    status_words = ["out", "questionable", "probable", "available", "doubtful"]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    for i, ln in enumerate(lines):
        low = ln.lower()
        if any(p.lower() in low for p in patterns):
            window = " | ".join(lines[i:i+3]).lower()
            for sw in status_words:
                if re.search(rf"\b{sw}\b", window):
                    return sw.upper()
    return None

def injury_overlay(player_name: str):
    out = {"status": None, "pdf_used": None, "note": None}
    pdf_url = find_latest_injury_pdf()
    if not pdf_url:
        out["note"] = "Could not find latest official NBA injury PDF."
        return out

    out["pdf_used"] = pdf_url
    ok, err = ensure_pypdf()
    if not ok:
        out["note"] = f"Could not install/read pypdf: {err}"
        return out

    try:
        from pypdf import PdfReader
        import io
        pdf_bytes = http_get_bytes(pdf_url)
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = "\n".join([(p.extract_text() or "") for p in reader.pages])
        out["status"] = parse_injury_status_from_text(text, player_name)
        if out["status"] is None:
            out["note"] = "Player not found on latest injury report PDF."
        return out
    except Exception as e:
        out["note"] = f"PDF parse failed safely: {e}"
        return out

# ---------------- Availability context ----------------

def load_injury_status(player_name: str):
    injuries_path = ART_DIR / "injuries_latest.csv"
    empty = {"status": None, "report_datetime": None, "team_abbr": None, "reason": None}
    if not injuries_path.exists():
        return empty
    try:
        injuries = pd.read_csv(injuries_path)
    except Exception:
        return empty
    for col in ["player", "status", "team_abbr", "reason", "report_datetime"]:
        if col not in injuries.columns:
            injuries[col] = ""
    injuries["player"] = injuries["player"].astype(str)
    injuries["status"] = injuries["status"].astype(str).str.upper()
    injuries["team_abbr"] = injuries["team_abbr"].astype(str)
    injuries["reason"] = injuries["reason"].astype(str)
    injuries["report_datetime"] = injuries["report_datetime"].astype(str)

    pq = player_name.strip().lower()
    match = injuries[injuries["player"].str.lower() == pq]
    if match.empty:
        match = injuries[injuries["player"].str.lower().str.contains(pq, na=False)]
    if match.empty:
        return empty
    row = match.iloc[0]
    return {
        "status": row.get("status") or None,
        "report_datetime": row.get("report_datetime") or None,
        "team_abbr": row.get("team_abbr") or None,
        "reason": row.get("reason") or None,
    }

# ---------------- Market context ----------------

def load_market_context(player_name: str, stat: str):
    market = {
        "market_open_line": None,
        "market_open_odds": None,
        "market_early_move": None,
        "market_close_line": None,
    }
    moves_path = Path("data/lines/props_line_movement.csv")
    closing_path = Path("data/lines/sdi_props_closing.csv")
    norm_player = " ".join(str(player_name).strip().lower().split())
    stat_norm = stat.strip().lower()

    if moves_path.exists():
        try:
            moves = pd.read_csv(moves_path)
        except Exception:
            moves = pd.DataFrame()
        if not moves.empty:
            for col in ["game_date", "player", "stat", "open_line", "open_over_odds", "open_under_odds", "open_ts"]:
                if col not in moves.columns:
                    moves[col] = pd.NA
            moves["player_norm"] = moves["player"].astype(str).str.lower().str.split().str.join(" ")
            moves["stat_norm"] = moves["stat"].astype(str).str.lower()
            subset = moves[(moves["player_norm"] == norm_player) & (moves["stat_norm"] == stat_norm)]
            if not subset.empty:
                subset = subset.sort_values("open_ts" if "open_ts" in subset.columns else "game_date")
                row = subset.iloc[0]
                market["market_open_line"] = float(row.get("open_line")) if pd.notna(row.get("open_line")) else None
                market["market_open_odds"] = {
                    "over": row.get("open_over_odds"),
                    "under": row.get("open_under_odds"),
                }
                if len(subset) > 1:
                    second = subset.iloc[1]
                    if pd.notna(second.get("open_line")) and pd.notna(row.get("open_line")):
                        market["market_early_move"] = float(second.get("open_line") - row.get("open_line"))

    if closing_path.exists():
        try:
            closing = pd.read_csv(closing_path)
        except Exception:
            closing = pd.DataFrame()
        if not closing.empty:
            for col in ["player", "stat", "line"]:
                if col not in closing.columns:
                    closing[col] = pd.NA
            closing["player_norm"] = closing["player"].astype(str).str.lower().str.split().str.join(" ")
            closing["stat_norm"] = closing["stat"].astype(str).str.lower()
            subset = closing[(closing["player_norm"] == norm_player) & (closing["stat_norm"] == stat_norm)]
            if not subset.empty:
                row = subset.iloc[-1]
                market["market_close_line"] = float(row.get("line")) if pd.notna(row.get("line")) else None

    return market

# ---------------- Teammate context ----------------

def load_teammates_out(team_abbr: str, game_date: str, player_name: str):
    out_json_path = ART_DIR / "out_players_today.json"
    if out_json_path.exists():
        try:
            out_json = json.loads(out_json_path.read_text())
            out_list = out_json.get(str(team_abbr), [])
            return [p for p in out_list if str(p).strip().lower() != str(player_name).strip().lower()]
        except Exception:
            pass
    avail_path = Path("data/injuries/availability_by_game.csv")
    if not avail_path.exists():
        return []
    try:
        avail = pd.read_csv(avail_path)
    except Exception:
        return []
    for col in ["game_date", "team_abbr", "player", "is_out"]:
        if col not in avail.columns:
            avail[col] = "" if col != "is_out" else 0
    avail["game_date"] = pd.to_datetime(avail["game_date"], errors="coerce").dt.date.astype(str)
    avail["team_abbr"] = avail["team_abbr"].astype(str)
    avail["player"] = avail["player"].astype(str)
    avail["is_out"] = pd.to_numeric(avail["is_out"], errors="coerce").fillna(0).astype(int)
    subset = avail[
        (avail["team_abbr"] == str(team_abbr))
        & (avail["game_date"] == str(game_date))
        & (avail["is_out"] == 1)
        & (avail["player"].str.lower() != str(player_name).strip().lower())
    ]
    return subset["player"].dropna().astype(str).head(3).tolist()

# ---------------- With/without impacts ----------------

def _norm_name(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def load_with_without_impacts(team_abbr: str, player_name: str, teammates_out: list[str]):
    impacts_path = MODELS_DIR / "with_without_impacts.parquet"
    if not impacts_path.exists():
        return {
            "bumps": [],
            "note": "no with/without sample",
            "d_min_total": 0.0,
            "d_pts_pm_total": 0.0,
            "d_reb_pm_total": 0.0,
            "d_ast_pm_total": 0.0,
        }
    try:
        impacts = pd.read_parquet(impacts_path)
    except Exception:
        return {
            "bumps": [],
            "note": "no with/without sample",
            "d_min_total": 0.0,
            "d_pts_pm_total": 0.0,
            "d_reb_pm_total": 0.0,
            "d_ast_pm_total": 0.0,
        }
    if impacts.empty:
        return {
            "bumps": [],
            "note": "no with/without sample",
            "d_min_total": 0.0,
            "d_pts_pm_total": 0.0,
            "d_reb_pm_total": 0.0,
            "d_ast_pm_total": 0.0,
        }

    impacts["player_norm"] = impacts["player"].astype(str).str.lower().str.split().str.join(" ")
    impacts["teammate_norm"] = impacts["teammate"].astype(str).str.lower().str.split().str.join(" ")
    player_norm = _norm_name(player_name)
    team_mask = impacts["team_abbr"].astype(str) == str(team_abbr)
    impacts = impacts[team_mask & (impacts["player_norm"] == player_norm)]

    bumps = []
    d_min_total = 0.0
    d_pts_total = 0.0
    d_reb_total = 0.0
    d_ast_total = 0.0

    for teammate in teammates_out:
        t_norm = _norm_name(teammate)
        row = impacts[impacts["teammate_norm"] == t_norm]
        if row.empty:
            continue
        entry = row.iloc[0]
        d_min = float(entry.get("d_min", 0.0))
        d_pts = float(entry.get("d_pts_pm", 0.0))
        d_reb = float(entry.get("d_reb_pm", 0.0))
        d_ast = float(entry.get("d_ast_pm", 0.0))
        bumps.append(
            {
                "teammate_out": teammate,
                "d_min": d_min,
                "d_pts_pm": d_pts,
                "d_reb_pm": d_reb,
                "d_ast_pm": d_ast,
                "sample_sizes": {
                    "n_with": int(entry.get("n_with", 0)),
                    "n_without": int(entry.get("n_without", 0)),
                },
            }
        )
        d_min_total += d_min
        d_pts_total += d_pts
        d_reb_total += d_reb
        d_ast_total += d_ast

    note = "no with/without sample" if not bumps else None
    return {
        "bumps": bumps,
        "note": note,
        "d_min_total": d_min_total,
        "d_pts_pm_total": d_pts_total,
        "d_reb_pm_total": d_reb_total,
        "d_ast_pm_total": d_ast_total,
    }

# ---------------- Prediction logic ----------------

def predict_player_prop(player: str, stat: str, line: float):
    feat_path = ART_DIR / "features_v1.parquet"
    if not feat_path.exists():
        raise FileNotFoundError("Missing artifacts/features_v1.parquet. Run Feature Factory first.")

    df = pd.read_parquet(feat_path)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["game_date"])

    last = find_last_row(df, player)

    cols_path = ART_DIR / "feature_cols_v2.json"
    min_model_path = ART_DIR / "xgb_min.joblib"
    rate_model_path = ART_DIR / f"xgb_{stat}_rate.joblib"
    if not (cols_path.exists() and min_model_path.exists() and rate_model_path.exists()):
        raise FileNotFoundError("Missing v2 artifacts. Run training step.")

    feature_cols = get_feature_cols(cols_path)
    min_model = get_model(min_model_path)
    rate_model = get_model(rate_model_path)

    row = {c: float(last.get(c, 0.0)) for c in feature_cols}
    X = pd.DataFrame([row]).apply(pd.to_numeric, errors="coerce").fillna(0.0)

    min_pred = float(np.clip(min_model.predict(X)[0], 0.0, 48.0))
    X_rate = X.copy()
    X_rate["min_pred_feature"] = min_pred
    rate_pred = float(np.clip(rate_model.predict(X_rate)[0], 0.0, 10.0))

    recent_minutes_roll = float(last.get("min_r5", last.get("min_r10", 0.0)))
    rotation_out_count = float(last.get("rotation_out_count", 0.0))
    starters_out_count = float(last.get("starters_out_count", 0.0))
    role = "starter" if max(recent_minutes_roll, min_pred) >= 26 else "bench"
    rotation_bump = rotation_out_count * (0.6 if role == "starter" else 0.35)
    rotation_bump += starters_out_count * (0.2 if role == "starter" else 0.4)

    teammates_out = load_teammates_out(
        str(last.get("team_abbr", "")),
        str(pd.to_datetime(last["game_date"]).date()),
        str(last["player"]),
    )
    impacts = load_with_without_impacts(str(last.get("team_abbr", "")), str(last["player"]), teammates_out)

    d_min_total = float(np.clip(impacts["d_min_total"], -6.0, 6.0))
    d_rate_pm_total = impacts["d_pts_pm_total"]
    if stat == "reb":
        d_rate_pm_total = impacts["d_reb_pm_total"]
    elif stat == "ast":
        d_rate_pm_total = impacts["d_ast_pm_total"]
    elif stat == "pra":
        d_rate_pm_total = impacts["d_pts_pm_total"] + impacts["d_reb_pm_total"] + impacts["d_ast_pm_total"]

    # Sharp Usage Redistribution Logic
    # We calculate how much the player's Usage Rate is expected to increase
    base_usg = float(last.get("usg_r10", 15.0))
    if base_usg < 1.0: base_usg = 15.0 # fallback

    # d_usg from historical with/without analysis
    d_usg_total = 0.0
    for b in impacts["bumps"]:
        d_usg_total += float(b.get("d_usg", 0.0))

    # Clip usg bump to avoid crazy projections
    d_usg_total = float(np.clip(d_usg_total, -5.0, 8.0))
    usg_adj_factor = (base_usg + d_usg_total) / base_usg

    rate_pct = d_rate_pm_total / rate_pred if rate_pred > 1e-6 else 0.0
    # Use a blend of empirical bump and theoretical usage redistribution
    if stat == "pts":
        # For points, USG redistribution is very relevant
        rate_pct = 0.5 * rate_pct + 0.5 * (usg_adj_factor - 1.0)

    rate_pct = float(np.clip(rate_pct, -0.15, 0.15))

    minutes_base = min_pred
    minutes_adj = float(np.clip(minutes_base + rotation_bump + d_min_total, 0.0, 44.0))
    rate_adj = rate_pred * (1.0 + rate_pct)

    pace_roll = float(last.get("team_pace_roll10", last.get("team_pace", 0.0)))
    opp_pace = float(last.get("opp_pace", 0.0))
    pace_diff = opp_pace - pace_roll
    shot_profile_adj = 0.0
    if stat == "pts":
        three_share = float(last.get("three_att_share_10g", 0.0))
        if pace_diff > 2.0 and three_share >= 0.4:
            shot_profile_adj = min(0.03, pace_diff / 50.0)
    rate_adj = rate_adj * (1.0 + shot_profile_adj)

    proj = minutes_adj * rate_adj

    if stat in ["reb", "ast", "pra", "stl", "blk", "tpm"]:
        # Sharp Poisson approach for discrete count stats
        # Poisson handles the skew and discrete nature of these stats better than Normal
        p_over = float(1.0 - poisson.cdf(line, proj))
        p_under = float(poisson.cdf(line - 0.001, proj))
        rmse = math.sqrt(proj) # Poisson variance is the mean
    else:
        # Normal approximation for higher-count stats (pts)
        rmse = load_rmse_v2(stat)
        z = (line - proj) / rmse if rmse > 1e-9 else 0.0
        p_over = 1.0 - normal_cdf(z)
        p_under = 1.0 - p_over

    best_p = max(p_over, p_under)
    best_side = "PASS"
    if best_p >= 0.55:
        best_side = "OVER" if p_over >= p_under else "UNDER"

    inj = injury_overlay(str(last["player"]))
    availability_injury = load_injury_status(str(last["player"]))
    market_ctx = load_market_context(str(last["player"]), stat)
    edge_open = None
    edge_close = None
    if market_ctx.get("market_open_line") is not None:
        edge_open = float(proj - market_ctx["market_open_line"])
    if market_ctx.get("market_close_line") is not None:
        edge_close = float(proj - market_ctx["market_close_line"])

    opp_def_rating_roll = float(last.get("opp_def_rating_roll", last.get("opp_drtg_roll10", 0.0)))

    if (availability_injury.get("status") and availability_injury.get("status").upper() == "OUT") or \
       (inj.get("status") and inj.get("status").upper() == "OUT"):
        best_side = "NO_BET"

    edge = float(proj - float(line))

    out = {
        "model_version": "v2_minutes_x_rate + matchup_merge + availability_context + injury_overlay",
        "player_query": player,
        "matched_player": str(last["player"]),
        "stat": stat,
        "line": float(line),
        "projection": float(proj),
        "edge": edge,
        "minutes_pred": float(minutes_base),
        "minutes_projection": float(minutes_adj),
        "rate_pred": float(rate_pred),
        "rate_adj": float(rate_adj),
        "rmse_used": float(rmse),
        "p_over": float(p_over),
        "p_under": float(p_under),
        "recommendation": best_side,
        "confidence_tier": tier(best_p) if best_side != "PASS" else "PASS",
        "last_game_date_used": str(pd.to_datetime(last["game_date"]).date()),
        "key_context": {
            "teammates_out": teammates_out,
            "opp_def_rating_roll": opp_def_rating_roll,
            "pace_roll": pace_roll,
            "recent_minutes_roll": recent_minutes_roll,
        },
        "context": {
            "rest_days": float(last.get("rest_days", 0.0)),
            "b2b": int(last.get("b2b", 0)),
            "games_last_7d": float(last.get("games_last_7d", 0.0)),
            "team_drtg": float(last.get("team_drtg", 0.0)),
            "opp_drtg": float(last.get("opp_drtg", 0.0)),
            "team_pace": float(last.get("team_pace", 0.0)),
            "opp_pace": float(last.get("opp_pace", 0.0)),
        },
        "availability_context": {
            "team_out_count": float(last.get("team_out_count", 0.0)),
            "team_q_count": float(last.get("team_q_count", 0.0)),
            "opp_out_count": float(last.get("opp_out_count", 0.0)),
            "opp_q_count": float(last.get("opp_q_count", 0.0)),
            "top_teammate_out_flag": float(last.get("top_teammate_out_flag", 0.0)),
            "out_teammates_min_proxy": float(last.get("out_teammates_min_proxy", 0.0)),
            "injury_report_status": availability_injury,
        },
        "lineup_context": {
            "player_on_off_net": float(last.get("player_on_off_net", 0.0)),
            "player_on_off_pace": float(last.get("player_on_off_pace", 0.0)),
            "opp_def_net_recent": float(last.get("opp_def_net_recent", 0.0)),
            "synergy_delta_proxy": float(last.get("synergy_delta_proxy", 0.0)),
            "cache_timestamp_used": str(last.get("lineup_cache_timestamp", "")),
        },
        "market_context": {
            "market_open_line": market_ctx.get("market_open_line"),
            "market_open_odds": market_ctx.get("market_open_odds"),
            "market_early_move": market_ctx.get("market_early_move"),
            "market_close_line": market_ctx.get("market_close_line"),
            "edge_vs_open": edge_open,
            "edge_vs_close": edge_close,
        },
        "injury_overlay": inj,
        "with_without_context": {
            "minutes_base": float(minutes_base),
            "minutes_adj": float(minutes_adj),
            "minutes_role_adj": float(rotation_bump),
            "minutes_with_without_adj": float(d_min_total),
            "rate_pct_adj": float(rate_pct),
            "usage_base": float(base_usg),
            "usage_adj_delta": float(d_usg_total),
            "shot_profile_adj": float(shot_profile_adj),
            "role": role,
            "bumps_applied": impacts["bumps"],
            "note": impacts.get("note"),
        },
    }
    return out

# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", required=True)
    parser.add_argument("--stat", required=True, choices=["pts", "reb", "ast", "pra", "stl", "blk", "tpm"])
    parser.add_argument("--line", required=True, type=float)
    args = parser.parse_args()

    try:
        out = predict_player_prop(args.player, args.stat, args.line)

        ART_DIR.mkdir(parents=True, exist_ok=True)
        (ART_DIR / "player_pick.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))

    except Exception as e:
        write_error(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
