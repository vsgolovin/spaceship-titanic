import pandas as pd
import click


@click.command()
@click.argument("src", type=click.Path())
@click.argument("dst", type=click.Path())
def main(src: str, dst: str):
    """
    Transform string columns in `src` to columns of features.
    * `PassengerId` -> `pid_0`, `pid_1` (both int)
    * `Cabin` -> `Deck` (str), `Num` (int), `Side` (str)

    Also for some reason simplifies `Destination` column values.
    """
    df = pd.read_csv(src)
    df[["pid_0", "pid_1"]] = df["PassengerId"].str.split("_", expand=True) \
        .astype(int)  # not nans => can change type to drop leading zeros
    df[["Deck", "Num", "Side"]] = df["Cabin"].str.split("/", expand=True)
    # simplify destination planet names
    df["Destination"] = df["Destination"].map({
        "TRAPPIST-1e": "TRP",
        "PSO J318.5-22": "PSO",
        "55 Cancri e": "55C"
    })
    df.to_csv(dst, index=False)


if __name__ == "__main__":
    main()
