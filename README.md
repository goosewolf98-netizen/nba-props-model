# nba-props-model

## Running the Model

To generate a prediction for a player prop, use the provided `run_model.sh` script.

### Usage

```bash
./run_model.sh "<Player Name>" <stat> <line>
```

- **Player Name:** The full name of the player (e.g., "LeBron James"). Enclose in quotes.
- **Stat:** The statistic to predict. Options are `pts`, `reb`, or `ast`.
- **Line:** The sportsbook line you want to bet against (e.g., 25.5).

### Example

To check if you should bet OVER or UNDER 32.5 points for Luka Doncic:

```bash
./run_model.sh "Luka Doncic" pts 32.5
```

### Output

The script will output a JSON summary including:
- **Projected Value:** The model's predicted stat value.
- **Recommendation:** `OVER`, `UNDER`, or `PASS`.
- **Confidence Tier:** A/B/C/D ranking of the edge.
- **Injury/Context:** Relevant injury news and lineup changes.

## Prop lines input (optional)

To enable ROI and calibration backtests, provide a CSV at `data/lines/props_lines.csv`.

Required columns:
- `game_date` (YYYY-MM-DD)
- `player`
- `stat` (`pts`, `reb`, or `ast`)
- `line` (float)

Optional columns:
- `over_odds` (American odds, e.g., `-110` or `+105`)
- `under_odds`
- `book` (string)

If the file is missing or incomplete, the backtests will be skipped safely and the pipeline will still complete.
