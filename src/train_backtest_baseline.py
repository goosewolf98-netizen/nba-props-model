from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
ART_DIR = Path("artifacts")

def find_data_file() -> Path:
    files = sorted(RAW_DIR.glob("*.csv")) + sorted(RAW_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError("No data found in data/raw. Download step did not create a file.")
    return files[0]

if __name__ == "__main__":
    ART_DIR.mkdir(parents=True, exist_ok=True)

    data_path = find_data_file()
    print("FOUND DATA FILE:", data_path)

    # Just to prove the pipeline works end-to-end:
    if data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
    else:
        df = pd.read_parquet(data_path)

    print("DATA SHAPE:", df.shape)
    print("COLUMNS (first 40):", list(df.columns)[:40])

    # Save a small “proof” output so artifacts exist
    (ART_DIR / "pipeline_ok.txt").write_text(
        f"OK - loaded {data_path.name}\nrows={df.shape[0]} cols={df.shape[1]}\n"
    )
    print("WROTE artifacts/pipeline_ok.txt")
