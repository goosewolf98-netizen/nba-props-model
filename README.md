# nba-props-model

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
