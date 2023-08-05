import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.decomposition import KernelPCA
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from src.transforms import InteractionTransformer

import click


@click.group()
def cli():
    pass


@cli.command()
@click.option("--train-data", type=click.Path(),
              default="data/processed/train.csv")
@click.option("--test-data", type=click.Path(),
              default="")
def preprocess(train_data: str, test_data: str):
    df = pd.read_csv(train_data, index_col=0)
    df_out = preprocess_df(df)
    df_out.to_csv("data/processed/train_logreg.csv")

    # test data
    if test_data:
        df = pd.read_csv(test_data, index_col=0)
        df_out = preprocess_df(df)
        df_out.to_csv("data/processed/test_logreg.csv")


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    # separate dataframe for processed data
    data = pd.DataFrame(index=df.index)

    # one-hot encoding
    for col in ["HomePlanet", "Destination", "Deck", "Side"]:
        dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False)
        data = data.join(dummies)

    # boolean features (use floats)
    data["CryoSleep"] = df["CryoSleep"].astype(float)
    data["Child"] = (df["Age"].fillna(100) < 13).astype(float)
    data["VIP"] = df["VIP"].fillna(False).astype(float)
    data["InGroup"] = df["InGroup"].astype(float)

    # interactions
    data["EuroChild"] = data["HomePlanet_Europa"] * data["Child"]
    data["MarsChild"] = data["HomePlanet_Mars"] * data["Child"]

    # bills
    bill_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    data = data.join(df[bill_cols].fillna(0) / 1000.)
    log_bill_cols = ["log" + col for col in bill_cols]
    data[log_bill_cols] = np.log(data[bill_cols] + 1)
    return data


def get_data_pipeline(feature_names: list[str]) -> Pipeline:
    # fill CrySleep with mean + PCA
    bill_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    log_bill_cols = ["log" + col for col in bill_cols]
    ct = ColumnTransformer(
        transformers=[
            ("fill_cryo", SimpleImputer(strategy="mean"), ["CryoSleep"]),
            ("pca", KernelPCA(n_components=1, kernel="rbf"), log_bill_cols),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    # feature interactions with CryoSleep
    ohe_cols = []
    for prefix in ["HomePlanet", "Destination", "Deck"]:
        ohe_cols.extend([c for c in feature_names if c.startswith(prefix)])
    pairs = [("CryoSleep", col) for col in ohe_cols]
    inter_transform = InteractionTransformer(pairs)

    return Pipeline([
        ("cryo+pca", ct),
        ("interactions", inter_transform)
    ])


def get_preprocessed_data():
    df = pd.read_csv("data/processed/train_logreg.csv", index_col=0)
    labels = pd.read_csv("data/processed/labels.csv", index_col=0)
    return df, labels.to_numpy().flatten()


@cli.command()
@click.option("--n-splits", type=int, default=10)
def validation_curve(n_splits: int):
    df, y = get_preprocessed_data()
    data_pipe = get_data_pipeline(df.columns.to_list())

    # sweep parameters
    C_range = np.logspace(-2.5, 2, 20)
    skf = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True)
    train_scores = np.zeros((len(C_range), n_splits))
    test_scores = np.zeros_like(train_scores)

    # split and process data
    # outer loop is over splits, so that we don't repeat expensive processing
    for j, (train_idx, test_idx) in tqdm(enumerate(skf.split(df, y))):
        X_train = data_pipe.fit_transform(df.iloc[train_idx, :])
        X_test = data_pipe.transform(df.iloc[test_idx, :])

        # fit the model
        for i, C_i in enumerate(C_range):
            model = LogisticRegression(C=C_i, penalty="l1", solver="liblinear")
            model.fit(X_train, y[train_idx])
            train_scores[i, j] = model.score(X_train, y[train_idx])
            test_scores[i, j] = model.score(X_test, y[test_idx])

    # plot validation curve
    plt.figure()
    colors = iter("bg")
    for s, subset in zip((train_scores, test_scores), ("train", "test")):
        mu = s.mean(1)
        sigma = s.std(1)
        c = next(colors)
        plt.semilogx(C_range, mu, color=c, marker="o", label=subset)
        plt.fill_between(C_range, mu - sigma, mu + sigma, color=c, alpha=0.15)
    plt.xlabel(r"$C$")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


@cli.command()
@click.option("--n-splits", type=int, default=5)
def learning_curve(n_splits: int):
    df, labels = get_preprocessed_data()
    data_pipeline = get_data_pipeline(df.columns.to_list())
    model = LogisticRegression(C=0.3, penalty="l1", solver="liblinear")
    pipeline = Pipeline(
        [("data", data_pipeline), ("logreg", model)]
    )
    trn_sizes, trn_scores, tst_scores = model_selection.learning_curve(
        estimator=pipeline,
        X=df,
        y=labels,
        train_sizes=np.linspace(0.1, 1.0, 12),
        cv=n_splits,
        n_jobs=4
    )
    plt.figure()
    colors = iter("bg")
    for scores, subset in zip([trn_scores, tst_scores], ("train", "test")):
        mu = scores.mean(1)
        sigma = scores.std(1)
        color = next(colors)
        plt.plot(trn_sizes, mu, color=color, marker='o', label=subset)
        plt.fill_between(trn_sizes, mu - sigma, mu + sigma, color=color,
                         alpha=0.15)
    plt.xlabel("Samples")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


@cli.command()
def train():
    df, labels = get_preprocessed_data()
    data_pipeline = get_data_pipeline(df.columns.to_list())
    model = LogisticRegression(C=0.3, penalty="l1", solver="liblinear")
    full_pipeline = Pipeline(
        [("data", data_pipeline), ("logreg", model)]
    )
    scores = model_selection.cross_val_score(
        estimator=full_pipeline,
        X=df,
        y=labels,
        cv=10,
        n_jobs=4
    )
    print(f"Accuracy: {scores.mean()} +- {scores.std()}")
    full_pipeline.fit(df, labels)
    for name, coef in zip(df.columns, full_pipeline[-1].coef_.ravel()):
        print(f"{name}: {coef:.3f}")
    with open("models/logreg.pkl", "wb") as fout:
        pickle.dump(full_pipeline, fout)


@cli.command()
def create_submission():
    df = pd.read_csv("data/processed/test_logreg.csv", index_col=0)
    with open("models/logreg.pkl", "rb") as fin:
        lr_pipeline = pickle.load(fin)
    predictions = lr_pipeline.predict(df)
    df_out = pd.DataFrame(index=df.index)
    df_out["Transported"] = predictions.astype(bool)
    df_out.to_csv("submission.csv")


if __name__ == "__main__":
    cli()
