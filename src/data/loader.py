import pandas as pd
from pathlib import Path
from typing import Optional

def load_csv_file(path: str, date_col: Optional[str] = None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found")
    if date_col:
        df = pd.read_csv(p, parse_dates=[date_col])
        df = df.set_index(date_col).sort_index()
    else:
        df = pd.read_csv(p)
    # keep only numeric columns (commodities)
    df_num = df.select_dtypes(include=['number']).copy()
    if df_num.shape[1] == 0:
        raise ValueError("No numeric columns found in the CSV. Check your file.")
    return df_num

if __name__ == "__main__":
    import sys
    import os
    project_root = r"C:\Users\shiva\Documents\multicommodity-prices-indonesia"
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\shiva\Documents\multicommodity-prices-indonesia\data\csv\india_commodities_demo.csv"
    df = load_csv_file(path)
    print("Loaded data shape:", df.shape)
    print("Columns:", df.columns.tolist())
