from pathlib import Path
import pandas as pd
from src.data import parse_str_cols, fill_group_nans

INP_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

for subset in ("train", "test"):
    df = pd.read_csv(INP_DIR / f"{subset}.csv")
    parse_str_cols(df)
    fill_group_nans(df)
    if subset == "train":
        labels = df.loc[:, "Transported"]
        labels.to_csv(OUT_DIR / "labels.csv")
        df.drop(columns=["Transported"], inplace=True)
    df.to_csv(OUT_DIR / f"{subset}.csv")
