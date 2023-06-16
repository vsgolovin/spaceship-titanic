from pathlib import Path
import numpy as np
import pandas as pd
import click


@click.command()
@click.argument("src", type=click.Path())
@click.argument("dst", type=click.Path())
def main(src: str, dst: str):
    df = pd.read_csv(src)

    # separate dataframe for processed data
    data = pd.DataFrame(index=df.index)
    data = data.join(pd.get_dummies(df["HomePlanet"], prefix="HP"))
    data["CryoSleep"] = df["CryoSleep"].astype(float)
    data["CryoSleep"].fillna(data["CryoSleep"].mean(), inplace=True)
    data["Age"] = df["Age"].fillna(df["Age"].median()) / 80.
    data["VIP"] = df["VIP"].fillna(False).astype(int)
    for col in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]:
        data["log_" + col] = np.log(df[col].fillna(0) + 1.)
    for col in ["Deck", "Side", "Destination"]:
        data = data.join(pd.get_dummies(df[col], prefix=col))
    group_pids = df.loc[df["pid_1"] != "01", "pid_0"].unique()
    data["in_group"] = df["pid_0"].isin(group_pids).astype(int)

    # create numpy array of features
    X = data.iloc[:, :].to_numpy(dtype=np.float64)
    np.save(dst, X, allow_pickle=False)

    # save labels if processing train set
    label_col = "Transported"
    if label_col in df.columns:
        outfile = Path(dst).parent / "labels.npy"
        np.save(outfile, df[label_col].to_numpy(dtype=int))


if __name__ == "__main__":
    main()
