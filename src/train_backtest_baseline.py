from pathlib import Path
import json
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

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
    drop = {"player", "team", "opp", "game_date", "pts", "reb", "ast", "min",
            "pts_rate", "reb_rate", "ast_rate"}
    cols = []
    for c in df.columns:
        if c in drop:
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

    r_model = _fit_xgb(train_rate[feature_cols], train_rate[rate_col])

    # Predict
    min_pred = m_model.predict(test_df[feature_cols])
    rate_pred = r_model.predict(test_df[feature_cols])

    # sanity clips
    min_pred = np.clip(min_pred, 0.0, 48.0)
    rate_pred = np.clip(rate_pred, 0.0, 10.0)

    proj = min_pred * rate_pred
    return m_model, r_model, min_pred, rate_pred, proj


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

        out = test_df[["game_date", "player", "team", "opp"]].copy()
        out["stat"] = stat
        out["y_true"] = y_true
        out["min_pred"] = min_pred
        out["rate_pred"] = rate_pred
        out["y_pred"] = proj
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

    for stat in ["pts", "reb", "ast"]:
        rate_col = f"{stat}_rate"
        rate_df = df[df["min"] >= MIN_MINUTES_FOR_RATE_TRAIN]
        if len(rate_df) < 200:
            rate_df = df
        r_model = _fit_xgb(rate_df[feature_cols], rate_df[rate_col])
        dump(r_model, ART_DIR / f"xgb_{stat}_rate.joblib")

    # --- Walk-forward v2 ---
    wf_all = {"settings": {
        "MIN_TRAIN_DAYS": MIN_TRAIN_DAYS,
        "HORIZON_DAYS": HORIZON_DAYS,
        "STEP_DAYS": STEP_DAYS,
        "MAX_FOLDS": MAX_FOLDS,
        "MIN_MINUTES_FOR_RATE_TRAIN": MIN_MINUTES_FOR_RATE_TRAIN,
    }}

    wf_preds_all = []
    for stat in ["pts", "reb", "ast"]:
        wf_metrics, wf_preds = walk_forward_v2(df, stat, feature_cols)
        wf_all[stat] = wf_metrics
        if len(wf_preds) > 0:
            wf_preds_all.append(wf_preds)

    (ART_DIR / "walkforward_metrics_v2.json").write_text(json.dumps(wf_all, indent=2))
    if wf_preds_all:
        pd.concat(wf_preds_all, ignore_index=True).to_csv(ART_DIR / "walkforward_predictions_v2.csv", index=False)

    print("Saved v2 artifacts: backtest_metrics_v2.json, walkforward_metrics_v2.json, walkforward_predictions_v2.csv, xgb_min.joblib, xgb_*_rate.joblib, feature_cols_v2.json")
