from pathlib import Path
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

RAW_DIR = Path("data/raw")
ART_DIR = Path("artifacts")

def find_data_file() -> Path:
    files = sorted(RAW_DIR.glob("*.csv")) + sorted(RAW_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError("No data found in data/raw. Download step did not create a file.")
    return files[0]

def load_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    # exact match
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]

    # contains match
    for c in cols:
        cl = c.lower()
        for cand in candidates:
            if cand in cl:
                return c
    return None

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Broad auto-detect across common boxscore schemas
    player_c = pick_col(df, ["player", "player_name", "athlete", "athlete_name", "athlete_display_name", "name"])
    date_c   = pick_col(df, ["game_date", "date", "start_date", "game_datetime"])
    min_c    = pick_col(df, ["min", "minutes", "mp"])
    pts_c    = pick_col(df, ["pts", "points"])
    reb_c    = pick_col(df, ["reb", "rebounds", "trb", "rebs", "total_rebounds"])
    ast_c    = pick_col(df, ["ast", "assists"])

    missing = [k for k,v in {
        "player": player_c, "game_date": date_c, "min": min_c,
        "pts": pts_c, "reb": reb_c, "ast": ast_c
    }.items() if v is None]

    if missing:
        print("Available columns:", list(df.columns))
        raise ValueError(f"Could not find required columns: {missing}")

    df = df.rename(columns={
        player_c: "player",
        date_c: "game_date",
        min_c: "min",
        pts_c: "pts",
        reb_c: "reb",
        ast_c: "ast",
    }).copy()

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["game_date"])

    # Ensure numeric
    for c in ["min", "pts", "reb", "ast"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player", "game_date"]).copy()
    grp = df.groupby("player", group_keys=False)

    # rolling history (shift so we never leak current game)
    for stat in ["min", "pts", "reb", "ast"]:
        df[f"{stat}_r5"]   = grp[stat].shift(1).rolling(5,  min_periods=1).mean()
        df[f"{stat}_r10"]  = grp[stat].shift(1).rolling(10, min_periods=1).mean()
        df[f"{stat}_sd10"] = grp[stat].shift(1).rolling(10, min_periods=2).std()

    # simple “in-rotation” proxy
    df["gp_last14"] = grp["game_date"].shift(1).rolling(14, min_periods=1).count()

    return df.fillna(0)

def time_split(df: pd.DataFrame, train_frac: float = 0.75):
    df = df.sort_values("game_date")
    split_idx = int(len(df) * train_frac)
    return df.iloc[:split_idx], df.iloc[split_idx:]

def train_one(df: pd.DataFrame, target: str):
    feature_cols = [c for c in df.columns if c.endswith(("_r5", "_r10", "_sd10"))] + ["gp_last14"]
    train_df, test_df = time_split(df, 0.75)

    X_train, y_train = train_df[feature_cols], train_df[target]
    X_test, y_test   = test_df[feature_cols],  test_df[target]

    model = XGBRegressor(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.04,
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
    ART_DIR.mkdir(parents=True, exist_ok=True)

    data_path = find_data_file()
    print("Using data file:", data_path)

    df = load_df(data_path)
    print("Raw shape:", df.shape)

    df = standardize_columns(df)
    df = add_rolling_features(df)
    print("Feature shape:", df.shape)

    results = {}
    for target in ["pts", "reb", "ast"]:
        model, metrics = train_one(df, target)
        dump(model, ART_DIR / f"xgb_{target}.joblib")
        results[target] = metrics

    out = ART_DIR / "backtest_metrics.json"
    out.write_text(pd.Series(results).to_json(indent=2))
    print("Saved:", out)
    print("Metrics:", results)
