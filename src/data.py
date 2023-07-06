import pandas as pd


def parse_str_cols(df: pd.DataFrame) -> None:
    """
    Transform original dataframe (works inplace).
    * Extract numerical data from `PassengerId` and `Cabin`.
    * Simplify `Destination` values.
    """
    # parse PassengerId
    pid = df["PassengerId"].str.split("_", expand=True).astype(int)
    df["GroupId"] = pid.iloc[:, 0]
    counts = pid.iloc[:, 0].map(pid.iloc[:, 0].value_counts())
    df["InGroup"] = (counts > 1)
    # parse Cabin
    df[["Deck", "Num", "Side"]] = df["Cabin"].str.split("/", expand=True)
    # simplify destination planet names
    df["Destination"] = df["Destination"].map({
        "TRAPPIST-1e": "TRP",
        "PSO J318.5-22": "PSO",
        "55 Cancri e": "55C"
    })
    # drop Cabin and turn PasengerId into index
    df.drop(columns=["Cabin"], inplace=True)
    df.set_index("PassengerId", inplace=True)


def fill_group_nans(df: pd.DataFrame) -> None:
    "Fill missing values of `HomePlanet` and `Deck` for people in groups."
    def _fill_nans(x: pd.Series) -> pd.Series:
        is_na = x.isna()
        uniques = x.dropna().unique()
        if is_na.any() and len(uniques) == 1:
            return x.fillna(uniques[0])
        return x

    for col in ["HomePlanet", "Deck"]:
        df[col] = df.groupby("GroupId")[col].transform(_fill_nans)
