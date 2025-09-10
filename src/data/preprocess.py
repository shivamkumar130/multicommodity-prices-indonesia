\"\"\"preprocess.py - cleaning and normalization stub\"\"\"
import pandas as pd
def preprocess(df):
    return df.fillna(method='ffill')

if __name__ == \"__main__\":
    print(\"Run preprocessing from notebooks or call preprocess()\")
