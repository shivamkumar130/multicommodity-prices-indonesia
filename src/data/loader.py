\"\"\"loader.py - simple CSV loader\"\"\"
import pandas as pd
def load_csv(path):
    df = pd.read_csv(path)
    print(\"Loaded\", path, \"shape=\", df.shape)
    return df

if __name__ == \"__main__\":
    import sys
    p = sys.argv[1] if len(sys.argv)>1 else \"data/csv/pihps_2017-08_2021-07.csv\"
    load_csv(p)
