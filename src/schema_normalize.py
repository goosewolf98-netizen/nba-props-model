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


def norm_game_date(df: pd.DataFrame):
    if df is None or len(df) == 0:
        return df

    # Handle uppercase variants from nba_api
    for c in ["GAME_DATE", "DATE"]:
        if c in df.columns and "game_date" not in df.columns:
            df = df.rename(columns={c: "game_date"})

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date.astype(str)
    elif "date" in df.columns:
        df["game_date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
    return df


def norm_minutes(df: pd.DataFrame):
    if df is None or len(df) == 0:
        return df
    if "min" not in df.columns and "minutes" in df.columns:
        df = df.rename(columns={"minutes": "min"})
    return df


def norm_team_cols(df: pd.DataFrame):
    if df is None or len(df) == 0:
        return df
    if "team_abbr" not in df.columns:
        for c in ["team_abbreviation", "team", "abbr", "TEAM_ABBR", "TEAM_ABBREVIATION"]:
            if c in df.columns:
                df = df.rename(columns={c: "team_abbr"})
                break
    if "opp_abbr" not in df.columns:
        for c in ["opponent_team_abbreviation", "opponent_abbr", "opp", "opponent", "OPPONENT_TEAM_ABBREVIATION"]:
            if c in df.columns:
                df = df.rename(columns={c: "opp_abbr"})
                break

    if "opp_abbr" not in df.columns and "MATCHUP" in df.columns:
        # Matchup format is 'TEAM vs. OPP' or 'TEAM @ OPP'
        df["opp_abbr"] = df["MATCHUP"].str.split().str[-1]

    return df


def norm_all(df: pd.DataFrame):
    df = norm_game_date(df)
    df = norm_minutes(df)
    df = norm_team_cols(df)
    return df
