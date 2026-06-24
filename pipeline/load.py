import pandas as pd

from pipeline.config import STAGED_DIR

def save_staged(df, table_name):
    """Save cleaned table to data/clean/staged/."""
    path = STAGED_DIR / f"{table_name}.pkl"
    df.to_pickle(path)
    print(f"saved {table_name}: {df.shape} -> {path}")
    return path