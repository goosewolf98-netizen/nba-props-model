import pandas as pd


def _rename_first(df: pd.DataFrame, target: str, candidates) -> pd.DataFrame:
    if target in df.columns:
        return df
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        cand_lower = cand.lower()
        if cand_lower in cols_lower:
            return df.rename(columns={cols_lower[cand_lower]: target})
    return df


def normalize_team_abbr(df: pd.DataFrame) -> pd.DataFrame:
    df = _rename_first(
        df,
        "team_abbr",
        ["team_abbreviation", "team", "abbr", "teamAbbr", "TEAM_ABBR"],
    )
    df = _rename_first(
        df,
        "opp_abbr",
        [
            "opponent_team_abbreviation",
            "opponent_abbreviation",
            "opp_abbreviation",
            "oppAbbr",
            "OPP_ABBR",
        ],
    )
    return df


def normalize_game_date(df: pd.DataFrame) -> pd.DataFrame:
    df = _rename_first(df, "game_date", ["date"])
    if "game_date" not in df.columns:
        df["game_date"] = pd.NaT
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.normalize()
    return df


def safe_reindex(df: pd.DataFrame, cols, fill_map=None) -> pd.DataFrame:
    fill_map = fill_map or {}
    df = df.reindex(columns=cols)
    for col, val in fill_map.items():
        if col not in df.columns:
            df[col] = val
        else:
            df[col] = df[col].fillna(val)
    return df
