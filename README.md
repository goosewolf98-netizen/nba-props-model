# nba-props-model

## Prop lines input (optional)

To enable ROI and calibration backtests, provide a CSV at `data/lines/props_lines.csv`.

Required columns:
- `game_date` (YYYY-MM-DD)
- `player`
- `stat` (`pts`, `reb`, `ast`, `pra`, `stl`, `blk`, `tpm`)
- `line` (float)

Optional columns:
- `over_odds` (American odds or Decimal, e.g., `-110` or `1.91`)
- `under_odds`
- `book` (string)

## Professional Data Sourcing (Sharp Upgrades)

### 1. Real-Time Market Lines
The model now includes `src/fetch_live_odds_api.py`. To get real market info:
1.  Get a free API key from [The Odds API](https://the-odds-api.com/).
2.  Set it as a secret `THE_ODDS_API_KEY` in your GitHub repository.
3.  The `player-query` workflow will automatically fetch live lines.

### 2. Historical Lines (Backtesting)
Professional historical prop data is premium. To build your own archive:
- **SportsDataIO**: Use `src/sdi_props_lines.py` if you have a subscription.
- **Kaggle**: Search for `nba-odds-data` by `christophertreasure` or `basketball` by `wyattowalsh`.
- **Archive Growth**: Every time the `fetch_live_odds_api.py` script runs, it appends data to `data/lines/props_lines.csv`, allowing you to build your own real historical backtest set for free over time.

### 3. Poisson Probability
For discrete stats (REB, AST, STL, BLK, TPM), the model uses **Poisson distributions** instead of Normal approximations, ensuring sharp accuracy for low-count props.
