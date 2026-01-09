from pathlib import Path
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

RAW_DIR = Path("data/raw")

def find_data_file() -> Path:
    files = sorted(RAW_DIR.glob("*.csv")) + sorted(RAW_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            "No data file found in data/raw. The download step did not create a CSV."
        )
    return files[0]

def load_df(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}

    # 1) exact match
    for cand in candidates:
        if cand in lower:
            return lower[cand]

    # 2) contains match
    for c in cols:
        cl = c.lower()
        for cand in candidates:
            if cand in cl:
                return c
    return None

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Auto-detect common column names (no manual editing needed)
    player_c = pick_col(df, ["player", "player_name", "athlete", "athlete_name", "athlete_display_name", "name"])
    date_c   = pick_col(df, ["game_date", "date", "start_date"])
    min_c    = pick_col(df, ["min", "minutes", "mp"])
    pts_c    = pick_col(df, ["pts", "points"])
    reb_c    = pick_col(df, ["reb", "rebounds", "trb", "rebs"])
    ast_c    = pick_col(df, ["ast", "assists"])

    missing = [k for k,v in {
        "player": player_c, "game_date": date_c, "min": min_c,
        "pts": pts_c, "reb": reb_c, "ast": ast_c
    }.items() if v is None]

    if missing:
        # Print columns in the Actions log so we can fix fast if needed
        print("Available columns:", list(df.columns))
        raise ValueError(f"Could not find required columns: {missing}")

    df = df.rename(columns={
        player_c: "player",
        date_c: "game_date",
        min_c: "min",
        pts_c: "pts",
        reb_c: "reb",
        ast_c: "ast",
    })

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["game_date"])
    return df

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player", "game_date"]).copy()
    grp = df.groupby("player", group_keys=False)

    for stat in ["min", "pts", "reb", "ast"]:
        df[f"{stat}_r5"]  = grp[stat].shift(1).rolling(5,  min_periods=1).mean()
        df[f"{stat}_r10"] = grp[stat].shift(1).rolling(10, min_periods=1).mean()
        df[f"{stat}_sd10"]= grp[stat].shift(1).rolling(10, min_periods=2).std()

    df["gp_last14"] = grp["game_date"].shift(1).rolling(14, min_periods=1).count()
    return df.fillna(0)

def train_one(df: pd.DataFrame, target: str):
    features = [c for c in df.columns if c.endswith(("_r5", "_r10", "_sd10"))] + ["gp_last14"]

    df = df.sort_values("game_date")
    split_idx = int(len(df) * 0.75)
    train_df = df.iloc[:split_idx]
    test_df  = df.iloc[split_idx:]

    X_train, y_train = train_df[features], train_df[target]
    X_test,  y_test  = test_df[features],  test_df[target]

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
    print("Using data file:", data_path)

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
    print("Backtest metrics:", results)
