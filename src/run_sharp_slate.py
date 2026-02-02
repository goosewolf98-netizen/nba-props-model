import os
import subprocess
import sys
import pandas as pd
import json
from pathlib import Path

def run_cmd(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    print("=== NBA Sharp Slate Processor ===")

    # 1. Download latest data
    run_cmd([sys.executable, "src/download_nba_api_data.py"])

    # 2. Build features
    run_cmd([sys.executable, "src/feature_factory_v1.py"])

    # 3. Train/Refresh models (optional if you want to skip, but good for accuracy)
    # We use MAX_FOLDS=1 for a quick refresh or rely on existing models
    # run_cmd([sys.executable, "src/train_backtest_baseline.py"])

    # 4. Fetch live odds
    run_cmd([sys.executable, "src/fetch_live_odds_api.py"])

    # 5. Process all fetched lines
    lines_path = Path("data/lines/props_lines.csv")
    if not lines_path.exists():
        print("No live lines found. Make sure THE_ODDS_API_KEY is set.")
        return

    df_lines = pd.read_csv(lines_path)
    # Only process today's lines
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    df_today = df_lines[df_lines['game_date'] == today]

    if df_today.empty:
        print(f"No lines found for today ({today}).")
        # Try processing everything just in case
        df_today = df_lines

    print(f"Processing {len(df_today)} props...")

    results = []
    for _, row in df_today.iterrows():
        player = row['player']
        stat = row['stat']
        line = row['line']

        try:
            # We call the predict script and capture output
            cmd = [sys.executable, "src/predict_player_prop.py", "--player", player, "--stat", stat, "--line", str(line)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0:
                # The script prints JSON to stdout
                output = proc.stdout.strip()
                # Find the JSON part if there's other text
                if "{" in output:
                    json_str = output[output.find("{"):]
                    res = json.loads(json_str)
                    results.append(res)
            else:
                print(f"Failed to predict for {player} {stat}: {proc.stderr}")
        except Exception as e:
            print(f"Error processing {player} {stat}: {e}")

    if results:
        # Save summary
        summary_path = Path("artifacts/sharp_slate.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(results, indent=2))

        # Create a nice CSV overview
        flat_results = []
        for r in results:
            flat_results.append({
                "Player": r['matched_player'],
                "Stat": r['stat'],
                "Line": r['line'],
                "Proj": round(r['projection'], 2),
                "Edge": round(r['edge'], 2),
                "Over %": round(r['p_over'] * 100, 1),
                "Under %": round(r['p_under'] * 100, 1),
                "Pick": r['recommendation'],
                "Tier": r['confidence_tier']
            })

        df_res = pd.DataFrame(flat_results)
        # Filter for actual picks
        picks = df_res[df_res['Pick'].isin(['OVER', 'UNDER'])]
        picks = picks.sort_values(by="Edge", key=abs, ascending=False)

        picks_path = Path("artifacts/sharp_picks_today.csv")
        picks.to_csv(picks_path, index=False)

        print("\n=== TOP SHARP PICKS FOR TODAY ===")
        print(picks.head(20).to_string(index=False))
        print(f"\nFull report saved to {picks_path}")
    else:
        print("No predictions generated.")

if __name__ == "__main__":
    main()
