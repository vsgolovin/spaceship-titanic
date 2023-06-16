import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import click


@click.command()
@click.argument("feats", type=click.Path())
@click.argument("labels", type=click.Path())
@click.option("--model-type", type=click.Choice(["svm", "dtree", "xgbc"]),
              default="xgbc", show_default=True)
@click.option("--no-val", is_flag=True, default=False,
              help="do not perform validation, i.e., use all available data")
@click.option("--split-seed", type=int, default=42, show_default=True,
              help="seed used for train-validation split")
@click.option("--val-size", type=float, default=0.25, show_default=True,
              help="validation set size")
@click.option("--save-to", type=click.Path(), default="/dev/null",
              help="file to save pickled model to")
def main(feats: str, labels: str, model_type: str, no_val: bool,
         split_seed: int, val_size: float, save_to: str):
    # load data
    X = np.load(feats)
    y = np.load(labels)
    if not no_val:
        X, X_val, y, y_val = train_test_split(X, y, random_state=split_seed,
                                              stratify=y, test_size=val_size)

    # train and evaluate model
    model = get_model(model_type)
    model.fit(X, y)
    if not no_val:
        pred = model.predict(X_val)
        acc = (pred == y_val).mean()
        print(f"Validation accuracy: {acc}")

    # save model
    if save_to != "/dev/null":
        with open(save_to, "wb") as fout:
            pickle.dump(model, fout)


def get_model(name: str):
    name = name.lower()
    if name == "svm":
        return SVC(C=3.0, kernel="rbf")
    if name == "dtree":
        return DecisionTreeClassifier(max_depth=6, min_samples_leaf=10)
    if name == "xgbc":
        return XGBClassifier(n_estimators=100, max_depth=3)
    raise ValueError(f"Unknown model {name}")


if __name__ == "__main__":
    main()
