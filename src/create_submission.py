from pathlib import Path
import pickle
import numpy as np
import click


@click.command()
@click.argument("feats", type=click.Path())
@click.argument("model", type=click.Path())
def main(feats: str, model: str):
    # load model
    with open(model, "rb") as fin:
        clf = pickle.load(fin)

    # load data
    data = np.load(feats)
    inp_dir = Path(feats).parent
    with open(inp_dir / "passenger_ids.txt", "r") as fin:
        pids = fin.read().split()

    # make predictions
    predictions = clf.predict(data).astype(bool)
    assert len(predictions) == len(pids)

    # save predictions
    with open("submission.csv", "w") as fout:
        fout.write("PassengerId,Transported\n")
        for pid, pred in zip(pids, predictions):
            fout.write(f"{pid},{pred}\n")


if __name__ == "__main__":
    main()
