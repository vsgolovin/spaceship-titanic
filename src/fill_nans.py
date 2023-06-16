import pandas as pd
import click


@click.command()
@click.argument("src", type=click.Path())
@click.argument("dst", type=click.Path())
def main(src: str, dst: str):
    "Fill missing values of `HomePlanet` and `Deck` for people in groups."
    df = pd.read_csv(src)
    for col in ["HomePlanet", "Deck"]:
        df[col] = df.groupby("pid_0")[col].transform(fill_group_nans)
    df.to_csv(dst, index=False)


def fill_group_nans(x: pd.Series) -> pd.Series:
    is_na = x.isna()
    uniques = x.dropna().unique()
    if is_na.any() and len(uniques) == 1:
        return x.fillna(uniques[0])
    return x


if __name__ == "__main__":
    main()
