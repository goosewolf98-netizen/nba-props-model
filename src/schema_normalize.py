import pandas as pd


def ensure_col(df: pd.DataFrame, target: str, variants) -> pd.DataFrame:
    if target in df.columns:
        return df
    for variant in variants:
        if variant in df.columns:
            return df.rename(columns={variant: target})
    return df


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    if "game_date" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "game_date"})
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date.astype(str)
    return df


def safe_cols(df: pd.DataFrame, cols, fill_zero_cols=None) -> pd.DataFrame:
    out = df.reindex(columns=cols)
    if fill_zero_cols:
        for col in fill_zero_cols:
            if col in out.columns:
                out[col] = out[col].fillna(0)
    return out
