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
3.  The **NBA Prediction** workflow will automatically fetch live lines.

### 2. Historical Lines (Backtesting)
Professional historical prop data is premium. To build your own archive:
- **SportsDataIO**: Use `src/sdi_props_lines.py` if you have a subscription.
- **Kaggle**: Search for `nba-odds-data` by `christophertreasure` or `basketball` by `wyattowalsh`.
- **Archive Growth**: Every time the `fetch_live_odds_api.py` script runs, it appends data to `data/lines/props_lines.csv`, allowing you to build your own real historical backtest set for free over time.

### 3. Poisson Probability
For discrete stats (REB, AST, STL, BLK, TPM), the model uses **Poisson distributions** instead of Normal approximations, ensuring sharp accuracy for low-count props.

## Data Infrastructure & Collection

The project now includes a SQLite database (`data/nba.db`) to store historical data persistently.

### Database Updates
To update the database with the latest stats, prop lines, and sentiment analysis:
```bash
python src/update_database.py
```
This script will:
1.  Download latest boxscores from NBA API and upsert to `boxscores` table.
2.  Ingest prop lines from `data/lines/props_lines.csv` to `prop_lines` table.
3.  Scrape Reddit (r/nba, r/sportsbook) for sentiment (requires valid network/auth) to `reddit_sentiment` table.

## How to Use the Model

### Method 1: GitHub Actions (Recommended)
The easiest way to use the model is through the **Actions** tab in this repository:
1.  Go to **Actions** -> **NBA Prediction**.
2.  Click **Run workflow**.
3.  Enter the **Player Name**, **Stat**, and **Betting Line** (e.g., LeBron James, pts, 24.5).
4.  The model will download the latest data, calculate "sharp" features, and output a detailed recommendation (OVER/UNDER/PASS).

### Method 2: Local Execution (CLI)
If you are running the model on your own machine:

1.  **Download & Prep Data:**
    ```bash
    python src/download_nba_api_data.py
    python src/feature_factory_v1.py
    ```

2.  **Train Models:** (Only needs to be done once per day)
    ```bash
    python src/train_backtest_baseline.py
    ```

3.  **Make a Prediction:**
    ```bash
    # Example: LeBron James PRA (Points+Rebounds+Assists) line of 34.5
    python src/predict_player_prop.py --player "LeBron James" --stat pra --line 34.5
    ```

4.  **Run Full Slate (Batch):**
    If you have an Odds API key, you can process every available prop for the day at once:
    ```bash
    export THE_ODDS_API_KEY='your_key_here'
    python src/run_sharp_slate.py
    ```
    This will generate a `artifacts/sharp_picks_today.csv` with the best edges found across the entire league.

### Supported Stats
The model is fully trained on:
- `pts` (Points)
- `reb` (Rebounds)
- `ast` (Assists)
- `pra` (Points + Rebounds + Assists)
- `stl` (Steals)
- `blk` (Blocks)
- `tpm` (Three Pointers Made)
