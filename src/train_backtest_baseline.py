from pathlib import Path
import json
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

ART_DIR = Path("artifacts")

def time_split(df: pd.DataFrame, train_frac: float = 0.75):
    df = df.sort_values("game_date")
    split_idx = int(len(df) * train_frac)
    return df.iloc[:split_idx], df.iloc[split_idx:]

def train_one(df: pd.DataFrame, target: str, feature_cols: list[str]):
    train_df, test_df = time_split(df, 0.75)
    X_train, y_train = train_df[feature_cols], train_df[target]
    X_test, y_test   = test_df[feature_cols],  test_df[target]

    model = XGBRegressor(
        n_estimators=900,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=2,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(mean_squared_error(y_test, preds) ** 0.5)
    return model, {"mae": mae, "rmse": rmse, "n_test": int(len(test_df))}

if __name__ == "__main__":
    ART_DIR.mkdir(parents=True, exist_ok=True)

    feat_path = ART_DIR / "features_v1.parquet"
    if not feat_path.exists():
        raise FileNotFoundError("Missing artifacts/features_v1.parquet. Feature Factory step must run first.")

    df = pd.read_parquet(feat_path)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["game_date"])

    drop = {"pts","reb","ast","player","team","opp","game_date"}
    feature_cols = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]

    # Save feature list so prediction uses the same columns
    (ART_DIR / "feature_cols.json").write_text(json.dumps(feature_cols, indent=2))

    results = {}
    for target in ["pts", "reb", "ast"]:
        model, metrics = train_one(df, target, feature_cols)
        dump(model, ART_DIR / f"xgb_{target}.joblib")
        results[target] = metrics

    (ART_DIR / "backtest_metrics.json").write_text(json.dumps(results, indent=2))
    print("Saved artifacts/backtest_metrics.json and artifacts/feature_cols.json")
