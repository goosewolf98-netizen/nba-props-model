from pathlib import Path
import json
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

ART_DIR = Path("artifacts")

# Walk-forward settings (kept small so GitHub Actions runs fast)
MIN_TRAIN_DAYS = 60      # require at least this many unique dates to start
HORIZON_DAYS = 7         # predict the next 7 days each fold
STEP_DAYS = 7            # move forward 7 days each fold
MAX_FOLDS = 30           # safety cap


def rmse(y_true, y_pred) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def time_split_by_date(df: pd.DataFrame, train_frac: float = 0.75):
    # Stable date split (not row split)
    dates = sorted(df["game_date"].dropna().unique())
    split_idx = int(len(dates) * train_frac)
    train_dates = set(dates[:split_idx])
    test_dates = set(dates[split_idx:])
    return df[df["game_date"].isin(train_dates)], df[df["game_date"].isin(test_dates)]


def fit_xgb(X_train, y_train):
    return XGBRegressor(
        n_estimators=700,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=2,
    ).fit(X_train, y_train)


def walk_forward(df: pd.DataFrame, target: str, feature_cols: list[str]):
    df = df.sort_values("game_date").copy()
    dates = sorted(df["game_date"].dropna().unique())
    if len(dates) < (MIN_TRAIN_DAYS + HORIZON_DAYS):
        return {"error": f"Not enough dates for walk-forward: {len(dates)}"}, pd.DataFrame()

    preds_all = []
    fold_metrics = []

    start_i = MIN_TRAIN_DAYS
    folds = 0
    i = start_i

    while i < len(dates) - 1 and folds < MAX_FOLDS:
        train_end = dates[i - 1]
        test_start = dates[i]
        test_end_idx = min(i + HORIZON_DAYS - 1, len(dates) - 1)
        test_end = dates[test_end_idx]

        train_df = df[df["game_date"] <= train_end]
        test_df = df[(df["game_date"] >= test_start) & (df["game_date"] <= test_end)]

        if len(test_df) == 0 or len(train_df) == 0:
            i += STEP_DAYS
            continue

        model = fit_xgb(train_df[feature_cols], train_df[target])
        y_pred = model.predict(test_df[feature_cols])

        fold_rmse = rmse(test_df[target], y_pred)
        fold_mae = float(mean_absolute_error(test_df[target], y_pred))

        fold_metrics.append({
            "target": target,
            "train_end": str(pd.to_datetime(train_end).date()),
            "test_start": str(pd.to_datetime(test_start).date()),
            "test_end": str(pd.to_datetime(test_end).date()),
            "n_test": int(len(test_df)),
            "rmse": fold_rmse,
            "mae": fold_mae
        })

        out = test_df[["game_date", "player", "team", "opp"]].copy()
        out["target"] = target
        out["y_true"] = test_df[target].values
        out["y_pred"] = y_pred
        out["train_end"] = train_end
        preds_all.append(out)

        folds += 1
        i += STEP_DAYS

    preds_df = pd.concat(preds_all, ignore_index=True) if preds_all else pd.DataFrame()

    if len(preds_df) == 0:
        return {"error": "No folds produced predictions."}, pd.DataFrame()

    overall = {
        "target": target,
        "folds": int(folds),
        "n_pred": int(len(preds_df)),
        "rmse": rmse(preds_df["y_true"], preds_df["y_pred"]),
        "mae": float(mean_absolute_error(preds_df["y_true"], preds_df["y_pred"])),
    }

    return {"overall": overall, "folds": fold_metrics}, preds_df


if __name__ == "__main__":
    ART_DIR.mkdir(parents=True, exist_ok=True)

    feat_path = ART_DIR / "features_v1.parquet"
    if not feat_path.exists():
        raise FileNotFoundError("Missing artifacts/features_v1.parquet. Feature Factory step must run first.")

    df = pd.read_parquet(feat_path)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["game_date"])

    drop = {"pts", "reb", "ast", "player", "team", "opp", "game_date"}
    feature_cols = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]

    # Save features list for prediction step
    (ART_DIR / "feature_cols.json").write_text(json.dumps(feature_cols, indent=2))

    # --- Standard backtest + save models ---
    results = {}
    for target in ["pts", "reb", "ast"]:
        train_df, test_df = time_split_by_date(df, 0.75)
        model = fit_xgb(train_df[feature_cols], train_df[target])
        preds = model.predict(test_df[feature_cols])

        metrics = {
            "mae": float(mean_absolute_error(test_df[target], preds)),
            "rmse": rmse(test_df[target], preds),
            "n_test": int(len(test_df)),
        }
        results[target] = metrics
        dump(model, ART_DIR / f"xgb_{target}.joblib")

    (ART_DIR / "backtest_metrics.json").write_text(json.dumps(results, indent=2))
    print("Saved artifacts/backtest_metrics.json")

    # --- Walk-forward backtest ---
    wf_all = {"settings": {
        "MIN_TRAIN_DAYS": MIN_TRAIN_DAYS,
        "HORIZON_DAYS": HORIZON_DAYS,
        "STEP_DAYS": STEP_DAYS,
        "MAX_FOLDS": MAX_FOLDS,
    }}

    wf_preds_all = []
    for target in ["pts", "reb", "ast"]:
        wf_metrics, wf_preds = walk_forward(df, target, feature_cols)
        wf_all[target] = wf_metrics
        if len(wf_preds) > 0:
            wf_preds_all.append(wf_preds)

    (ART_DIR / "walkforward_metrics.json").write_text(json.dumps(wf_all, indent=2))

    if wf_preds_all:
        wf_preds_df = pd.concat(wf_preds_all, ignore_index=True)
        wf_preds_df.to_csv(ART_DIR / "walkforward_predictions.csv", index=False)

    print("Saved walkforward artifacts.")
