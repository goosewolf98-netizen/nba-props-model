from pathlib import Path
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

def find_data_file() -> Path:
    raw = Path("data/raw")
    files = sorted(raw.glob("*.parquet")) + sorted(raw.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No data found in data/raw. Did the download step run?")
    return files[0]

def load_df(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("player", "athlete", "athlete_name", "name"):
            rename[c] = "player"
        elif cl in ("date", "game_date", "start_date"):
            rename[c] = "game_date"
        elif cl in ("minutes", "min", "mp"):
            rename[c] = "min"
        elif cl in ("points", "pts"):
            rename[c] = "pts"
        elif cl in ("rebounds", "reb", "trb", "rebs"):
            rename[c] = "reb"
        elif cl in ("assists", "ast"):
            rename[c] = "ast"
    df = df.rename(columns=rename)
    need = {"player", "game_date", "min", "pts", "reb", "ast"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after standardization: {missing}")
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player", "game_date"]).copy()
    grp = df.groupby("player", group_keys=False)

    for stat in ["min", "pts", "reb", "ast"]:
        df[f"{stat}_r5"] = grp[stat].shift(1).rolling(5, min_periods=1).mean()
        df[f"{stat}_r10"] = grp[stat].shift(1).rolling(10, min_periods=1).mean()
        df[f"{stat}_sd10"] = grp[stat].shift(1).rolling(10, min_periods=2).std()

    df["gp_last14"] = grp["game_date"].shift(1).rolling(14, min_periods=1).count()
    return df.fillna(0)

def train_one(df: pd.DataFrame, target: str):
    features = [c for c in df.columns if c.endswith(("_r5", "_r10", "_sd10"))] + ["gp_last14"]

    df = df.sort_values("game_date")
    split_idx = int(len(df) * 0.75)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=2,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(mean_squared_error(y_test, preds, squared=False))
    return model, {"mae": mae, "rmse": rmse, "n_test": int(len(test_df))}

if __name__ == "__main__":
    data_path = find_data_file()
    df = load_df(data_path)
    df = standardize_columns(df)
    df = add_rolling_features(df)

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for target in ["pts", "reb", "ast"]:
        model, metrics = train_one(df, target)
        dump(model, out_dir / f"xgb_{target}.joblib")
        results[target] = metrics

    (out_dir / "backtest_metrics.json").write_text(pd.Series(results).to_json(indent=2))
    print(results)
