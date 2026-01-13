import json
import math
import argparse
import re
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen, Request

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
    try:
        import pypdf  # noqa
        return True, None
    except Exception:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pypdf"])
            import pypdf  # noqa
            return True, None
        except Exception as e:
            return False, str(e)

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

# ---------------- Main ----------------

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

        cols_path = ART_DIR / "feature_cols_v2.json"
        min_model_path = ART_DIR / "xgb_min.joblib"
        rate_model_path = ART_DIR / f"xgb_{args.stat}_rate.joblib"
        if not (cols_path.exists() and min_model_path.exists() and rate_model_path.exists()):
            raise FileNotFoundError("Missing v2 artifacts. Run training step.")

        feature_cols = json.loads(cols_path.read_text())
        min_model = load(min_model_path)
        rate_model = load(rate_model_path)

        row = {c: float(last.get(c, 0.0)) for c in feature_cols}
        X = pd.DataFrame([row]).apply(pd.to_numeric, errors="coerce").fillna(0.0)

        min_pred = float(np.clip(min_model.predict(X)[0], 0.0, 48.0))
        rate_pred = float(np.clip(rate_model.predict(X)[0], 0.0, 10.0))
        proj = min_pred * rate_pred

        rmse = load_rmse_v2(args.stat)
        z = (args.line - proj) / rmse if rmse > 1e-9 else 0.0
        p_over = 1.0 - normal_cdf(z)
        p_under = 1.0 - p_over

        best_p = max(p_over, p_under)
        best_side = "PASS"
        if best_p >= 0.55:
            best_side = "OVER" if p_over >= p_under else "UNDER"

        inj = injury_overlay(str(last["player"]))

        out = {
            "model_version": "v2_minutes_x_rate + matchup_merge + injury_overlay",
            "player_query": args.player,
            "matched_player": str(last["player"]),
            "stat": args.stat,
            "line": float(args.line),
            "projection": float(proj),
            "minutes_pred": float(min_pred),
            "rate_pred": float(rate_pred),
            "rmse_used": float(rmse),
            "p_over": float(p_over),
            "p_under": float(p_under),
            "recommendation": best_side,
            "confidence_tier": tier(best_p) if best_side != "PASS" else "PASS",
            "last_game_date_used": str(pd.to_datetime(last["game_date"]).date()),
            "context": {
                "rest_days": float(last.get("rest_days", 0.0)),
                "b2b": int(last.get("b2b", 0)),
                "games_last_7d": float(last.get("games_last_7d", 0.0)),
                "team_drtg": float(last.get("team_drtg", 0.0)),
                "opp_drtg": float(last.get("opp_drtg", 0.0)),
                "team_pace": float(last.get("team_pace", 0.0)),
                "opp_pace": float(last.get("opp_pace", 0.0)),
            },
            "injury_overlay": inj
        }

        ART_DIR.mkdir(parents=True, exist_ok=True)
        (ART_DIR / "player_pick.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))

    except Exception as e:
        write_error(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
