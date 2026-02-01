from pathlib import Path
import subprocess
import sys
import json
import os
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from backtest_thresholds import run_backtest
from backtest_roi import run_backtest as run_roi_backtest
from calibration_report import run_calibration

ART_DIR = Path("artifacts")

# Keep this reasonable for GitHub Actions runtime
MIN_TRAIN_DAYS = 60
HORIZON_DAYS = 7
STEP_DAYS = 7
MAX_FOLDS = 20

MIN_MINUTES_FOR_RATE_TRAIN = 4.0  # ignore tiny-minute games for rate targets


def _rmse(y_true, y_pred) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def _time_split_by_date(df: pd.DataFrame, train_frac: float = 0.75):
    dates = sorted(df["game_date"].dropna().unique())
    split_idx = int(len(dates) * train_frac)
    train_dates = set(dates[:split_idx])
    test_dates = set(dates[split_idx:])
    return df[df["game_date"].isin(train_dates)], df[df["game_date"].isin(test_dates)]


def _fit_xgb(X_train, y_train):
    return XGBRegressor(
        n_estimators=550,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=2,
    ).fit(X_train, y_train)


def _make_rate_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mins = out["min"].clip(lower=1.0)
    out["pts_rate"] = out["pts"] / mins
    out["reb_rate"] = out["reb"] / mins
    out["ast_rate"] = out["ast"] / mins
    return out


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {
        "player", "team", "opp", "team_abbr", "opp_abbr",
        "game_date", "pts", "reb", "ast", "min",
        "pts_rate", "reb_rate", "ast_rate",
    }
    cols = []
    use_market = os.getenv("USE_MARKET_FEATURES", "0") == "1"
    for c in df.columns:
        if c in drop:
            continue
        if not use_market and (c.startswith("market_open_") or c.startswith("line_move_early_") or c.startswith("market_book_count_")):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _predict_minutes_rate_projection(train_df: pd.DataFrame, test_df: pd.DataFrame, stat: str, feature_cols: list[str]):
    # Minutes model (train on all rows)
    m_model = _fit_xgb(train_df[feature_cols], train_df["min"])

    # Rate model (train on >= MIN_MINUTES_FOR_RATE_TRAIN)
    rate_col = f"{stat}_rate"
    train_rate = train_df[train_df["min"] >= MIN_MINUTES_FOR_RATE_TRAIN]
    if len(train_rate) < 200:
        # fallback: train on all rows if filtered too small
        train_rate = train_df

    train_rate = train_rate.copy()
    train_rate["min_pred_feature"] = m_model.predict(train_rate[feature_cols])
    rate_features = feature_cols + ["min_pred_feature"]
    r_model = _fit_xgb(train_rate[rate_features], train_rate[rate_col])

    # Predict
    min_pred = m_model.predict(test_df[feature_cols])
    test_features = test_df[feature_cols].copy()
    test_features["min_pred_feature"] = min_pred
    rate_pred = r_model.predict(test_features[rate_features])

    # sanity clips
    min_pred = np.clip(min_pred, 0.0, 48.0)
    rate_pred = np.clip(rate_pred, 0.0, 10.0)

    proj = min_pred * rate_pred
    return m_model, r_model, min_pred, rate_pred, proj


def _availability_slices(preds_df: pd.DataFrame) -> dict:
    slices = {}
    if len(preds_df) == 0:
        return slices
    preds_df = preds_df.copy()
    preds_df["y_true"] = pd.to_numeric(preds_df["y_true"], errors="coerce").fillna(0.0)
    preds_df["y_pred"] = pd.to_numeric(preds_df["y_pred"], errors="coerce").fillna(0.0)

    if "team_out_count" in preds_df.columns:
        ge_mask = preds_df["team_out_count"] >= 2
        lt_mask = preds_df["team_out_count"] < 2
        if ge_mask.any():
            slices["team_out_count_ge_2"] = _rmse(preds_df.loc[ge_mask, "y_true"], preds_df.loc[ge_mask, "y_pred"])
        if lt_mask.any():
            slices["team_out_count_lt_2"] = _rmse(preds_df.loc[lt_mask, "y_true"], preds_df.loc[lt_mask, "y_pred"])

    if "top_teammate_out_flag" in preds_df.columns:
        on_mask = preds_df["top_teammate_out_flag"] == 1
        off_mask = preds_df["top_teammate_out_flag"] == 0
        if on_mask.any():
            slices["top_teammate_out_flag_1"] = _rmse(preds_df.loc[on_mask, "y_true"], preds_df.loc[on_mask, "y_pred"])
        if off_mask.any():
            slices["top_teammate_out_flag_0"] = _rmse(preds_df.loc[off_mask, "y_true"], preds_df.loc[off_mask, "y_pred"])

    if "player_on_off_net" in preds_df.columns:
        onoff = pd.to_numeric(preds_df["player_on_off_net"], errors="coerce").fillna(0.0).abs()
        high_mask = onoff >= 2.0
        low_mask = onoff < 2.0
        if high_mask.any():
            slices["player_on_off_net_ge_2"] = _rmse(preds_df.loc[high_mask, "y_true"], preds_df.loc[high_mask, "y_pred"])
        if low_mask.any():
            slices["player_on_off_net_lt_2"] = _rmse(preds_df.loc[low_mask, "y_true"], preds_df.loc[low_mask, "y_pred"])

    return slices


def walk_forward_v2(df: pd.DataFrame, stat: str, feature_cols: list[str]):
    df = df.sort_values("game_date").copy()
    dates = sorted(df["game_date"].dropna().unique())
    if len(dates) < (MIN_TRAIN_DAYS + HORIZON_DAYS):
        return {"error": f"Not enough dates for walk-forward: {len(dates)}"}, pd.DataFrame()

    preds_all = []
    fold_metrics = []

    i = MIN_TRAIN_DAYS
    folds = 0

    while i < len(dates) - 1 and folds < MAX_FOLDS:
        train_end = dates[i - 1]
        test_start = dates[i]
        test_end_idx = min(i + HORIZON_DAYS - 1, len(dates) - 1)
        test_end = dates[test_end_idx]

        train_df = df[df["game_date"] <= train_end]
        test_df = df[(df["game_date"] >= test_start) & (df["game_date"] <= test_end)]

        if len(train_df) == 0 or len(test_df) == 0:
            i += STEP_DAYS
            continue

        _, _, min_pred, rate_pred, proj = _predict_minutes_rate_projection(train_df, test_df, stat, feature_cols)

        y_true = test_df[stat].values
        fold_rmse = _rmse(y_true, proj)
        fold_mae = float(mean_absolute_error(y_true, proj))

        fold_metrics.append({
            "stat": stat,
            "train_end": str(pd.to_datetime(train_end).date()),
            "test_start": str(pd.to_datetime(test_start).date()),
            "test_end": str(pd.to_datetime(test_end).date()),
            "n_test": int(len(test_df)),
            "rmse": fold_rmse,
            "mae": fold_mae
        })

        extra_cols = [
            "team_out_count", "team_q_count", "opp_out_count", "opp_q_count",
            "top_teammate_out_flag", "out_teammates_min_proxy",
            "player_on_off_net", "player_on_off_pace", "opp_def_net_recent",
            "top_synergy_teammate_on_flag", "synergy_delta_proxy",
        ]
        cols = ["game_date", "player"]
        if "team_abbr" in test_df.columns:
            cols.append("team_abbr")
        elif "team" in test_df.columns:
            cols.append("team")
        else:
            test_df["team_abbr"] = ""
            cols.append("team_abbr")
        if "opp_abbr" in test_df.columns:
            cols.append("opp_abbr")
        elif "opp" in test_df.columns:
            cols.append("opp")
        else:
            test_df["opp_abbr"] = ""
            cols.append("opp_abbr")
        cols = cols + [c for c in extra_cols if c in test_df.columns]
        out = test_df[cols].copy()
        if "team_abbr" in out.columns:
            out = out.rename(columns={"team_abbr": "team"})
        if "opp_abbr" in out.columns:
            out = out.rename(columns={"opp_abbr": "opp"})
        out["stat"] = stat
        out["y_true"] = y_true
        out["min_pred"] = min_pred
        out["rate_pred"] = rate_pred
        out["y_pred"] = proj
        out["projection"] = proj
        out["actual"] = y_true
        out["p_over"] = np.nan
        out["p_under"] = np.nan
        out["train_end"] = train_end
        preds_all.append(out)

        folds += 1
        i += STEP_DAYS

    preds_df = pd.concat(preds_all, ignore_index=True) if preds_all else pd.DataFrame()
    if len(preds_df) == 0:
        return {"error": "No folds produced predictions."}, pd.DataFrame()

    overall = {
        "stat": stat,
        "folds": int(folds),
        "n_pred": int(len(preds_df)),
        "rmse": _rmse(preds_df["y_true"], preds_df["y_pred"]),
        "mae": float(mean_absolute_error(preds_df["y_true"], preds_df["y_pred"])),
    }

    return {"overall": overall, "folds": fold_metrics}, preds_df


if __name__ == "__main__":
    ART_DIR.mkdir(parents=True, exist_ok=True)

    feat_path = ART_DIR / "features_v1.parquet"
    if not feat_path.exists():
        raise FileNotFoundError("Missing artifacts/features_v1.parquet. Feature Factory must run first.")

    df = pd.read_parquet(feat_path)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["game_date"]).copy()

    # numeric coercions
    for c in ["min", "pts", "reb", "ast"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df = _make_rate_targets(df)
    feature_cols = _get_feature_cols(df)

    # Save feature columns for v2 prediction
    (ART_DIR / "feature_cols_v2.json").write_text(json.dumps(feature_cols, indent=2))

    # --- Holdout backtest v2 + save models trained on ALL data (for serving) ---
    train_df, test_df = _time_split_by_date(df, 0.75)

    back_v2 = {"settings": {
        "MIN_MINUTES_FOR_RATE_TRAIN": MIN_MINUTES_FOR_RATE_TRAIN
    }}

    # Evaluate each stat using minutes×rate projection
    for stat in ["pts", "reb", "ast"]:
        _, _, min_pred, rate_pred, proj = _predict_minutes_rate_projection(train_df, test_df, stat, feature_cols)
        back_v2[stat] = {
            "rmse": _rmse(test_df[stat].values, proj),
            "mae": float(mean_absolute_error(test_df[stat].values, proj)),
            "n_test": int(len(test_df)),
        }

    (ART_DIR / "backtest_metrics_v2.json").write_text(json.dumps(back_v2, indent=2))

    # Train serving models on ALL rows
    m_model = _fit_xgb(df[feature_cols], df["min"])
    dump(m_model, ART_DIR / "xgb_min.joblib")
    df = df.copy()
    df["min_pred_feature"] = m_model.predict(df[feature_cols])

    importance_rows = []
    if hasattr(m_model, "feature_importances_"):
        for feature, importance in zip(feature_cols, m_model.feature_importances_):
            importance_rows.append({"model": "min", "feature": feature, "importance": float(importance)})

    for stat in ["pts", "reb", "ast"]:
        rate_col = f"{stat}_rate"
        rate_df = df[df["min"] >= MIN_MINUTES_FOR_RATE_TRAIN]
        if len(rate_df) < 200:
            rate_df = df
        rate_df = rate_df.copy()
        rate_df["min_pred_feature"] = m_model.predict(rate_df[feature_cols])
        rate_features = feature_cols + ["min_pred_feature"]
        r_model = _fit_xgb(rate_df[rate_features], rate_df[rate_col])
        dump(r_model, ART_DIR / f"xgb_{stat}_rate.joblib")
        if hasattr(r_model, "feature_importances_"):
            for feature, importance in zip(feature_cols, r_model.feature_importances_):
                importance_rows.append({
                    "model": f"{stat}_rate",
                    "feature": feature,
                    "importance": float(importance),
                })

    if importance_rows:
        imp_df = pd.DataFrame(importance_rows)
        imp_df = imp_df.sort_values(["model", "importance"], ascending=[True, False])
        imp_df = imp_df.groupby("model", as_index=False).head(30)
        imp_df.to_csv(ART_DIR / "feature_importance_v2.csv", index=False)

    # --- Walk-forward v2 ---
    wf_all = {"settings": {
        "MIN_TRAIN_DAYS": MIN_TRAIN_DAYS,
        "HORIZON_DAYS": HORIZON_DAYS,
        "STEP_DAYS": STEP_DAYS,
        "MAX_FOLDS": MAX_FOLDS,
        "MIN_MINUTES_FOR_RATE_TRAIN": MIN_MINUTES_FOR_RATE_TRAIN,
    }}

    wf_preds_all = []
    wf_slices = {}
    try:
        for stat in ["pts", "reb", "ast"]:
            wf_metrics, wf_preds = walk_forward_v2(df, stat, feature_cols)
            if isinstance(wf_metrics, dict) and "overall" in wf_metrics and len(wf_preds) > 0:
                wf_metrics["availability_slices"] = _availability_slices(wf_preds)
                wf_slices[stat] = wf_metrics["availability_slices"]
            wf_all[stat] = wf_metrics
            if len(wf_preds) > 0:
                wf_preds_all.append(wf_preds)
    except Exception as exc:
        (ART_DIR / "walkforward_predictions_v2.csv").write_text(
            ",".join(["game_date", "player", "stat", "projection", "p_over", "p_under", "actual"]) + "\n"
        )
        (ART_DIR / "walkforward_metrics_v2.json").write_text(json.dumps({
            "status": "failed",
            "error": str(exc),
        }, indent=2))
        raise

    (ART_DIR / "walkforward_metrics_v2.json").write_text(json.dumps(wf_all, indent=2))
    pred_path = ART_DIR / "walkforward_predictions_v2.csv"
    required_cols = ["game_date", "player", "stat", "projection", "p_over", "p_under", "actual"]
    if wf_preds_all:
        preds_out = pd.concat(wf_preds_all, ignore_index=True)
        for col in required_cols:
            if col not in preds_out.columns:
                preds_out[col] = np.nan if col in ["p_over", "p_under"] else pd.NA
        preds_out.to_csv(pred_path, index=False)
    else:
        empty = pd.DataFrame(columns=required_cols)
        empty["p_over"] = np.nan
        empty["p_under"] = np.nan
        empty.to_csv(pred_path, index=False)
    try:
        subprocess.check_call([sys.executable, "src/sdi_props_lines.py"])
        subprocess.check_call([sys.executable, "src/sdi_props_closing.py"])
    except Exception as exc:
        print(f"SportsDataIO lines skipped due to error: {exc}")
    try:
        run_roi_backtest(pred_path, Path("data/lines/sdi_props_closing.csv"))
    except Exception as exc:
        print(f"ROI backtest skipped due to error: {exc}")
    try:
        run_calibration(pred_path, Path("data/lines/sdi_props_closing.csv"))
    except Exception as exc:
        print(f"Calibration report skipped due to error: {exc}")
    try:
        run_backtest(pred_path, ART_DIR)
    except Exception as exc:
        print(f"Threshold backtest skipped due to error: {exc}")

    (ART_DIR / "walkforward_slices_v2.json").write_text(json.dumps(wf_slices, indent=2))

    def run_bulk75():
        ART_DIR.mkdir(parents=True, exist_ok=True)
        lines_dir = Path("data/lines")
        lines_dir.mkdir(parents=True, exist_ok=True)

        wf_pred_path = ART_DIR / "walkforward_predictions_v2.csv"
        wf_metrics_path = ART_DIR / "walkforward_metrics_v2.json"
        required_cols = ["game_date", "player", "stat", "projection", "p_over", "p_under", "actual"]
        if not wf_pred_path.exists():
            empty = pd.DataFrame(columns=required_cols)
            empty["p_over"] = np.nan
            empty["p_under"] = np.nan
            empty.to_csv(wf_pred_path, index=False)
        if not wf_metrics_path.exists():
            wf_metrics_path.write_text(json.dumps({"status": "placeholder"}, indent=2))

        subprocess.run([sys.executable, "src/sdi_props_lines.py"], check=False)
        subprocess.run([sys.executable, "src/sdi_props_closing.py"], check=False)
        subprocess.run([sys.executable, "src/backtest_roi.py"], check=False)
        subprocess.run([sys.executable, "src/calibration_report.py"], check=False)

        roi_summary = ART_DIR / "roi_backtest_summary.json"
        if not roi_summary.exists():
            roi_summary.write_text(json.dumps({"status": "skipped", "reason": "no lines matched"}, indent=2))

        calib_report = ART_DIR / "calibration_report.json"
        if not calib_report.exists():
            calib_report.write_text(json.dumps({"status": "skipped", "reason": "no lines matched"}, indent=2))

    run_bulk75()

    print("Saved v2 artifacts: backtest_metrics_v2.json, walkforward_metrics_v2.json, walkforward_predictions_v2.csv, xgb_min.joblib, xgb_*_rate.joblib, feature_cols_v2.json")
