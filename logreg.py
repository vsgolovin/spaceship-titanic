import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.decomposition import KernelPCA
import click


@click.group()
def cli():
    pass


@cli.command()
@click.option("--train-data", type=click.Path(),
              default="data/processed/train.csv")
@click.option("--test-data", type=click.Path(),
              default="")
def process_data(train_data: str, test_data: str):
    pca = KernelPCA(n_components=1, kernel="rbf")

    # train data
    df = pd.read_csv(train_data, index_col=0)
    df_out = process_df(df, pca)
    df_out.to_csv("data/processed/X.csv")

    # test data
    if test_data:
        df = pd.read_csv(test_data, index_col=0)
        df_out = process_df(df, pca, True)
        df_out.to_csv("data/processed/X_test.csv")


def process_df(df: pd.DataFrame, pca: KernelPCA, is_test: bool = False):
    # separate dataframe for processed data
    data = pd.DataFrame(index=df.index)

    # one-hot encoding
    ohe_cols = []  # save for interactions
    for col in ["HomePlanet", "Destination", "Deck", "Side"]:
        dummies = pd.get_dummies(df[col], prefix=col)
        ohe_cols.extend(dummies.columns)
        data = data.join(dummies)

    # boolean features (use floats)
    data["CryoSleep"] = df["CryoSleep"].astype(float).fillna(0.358)  # mean
    data["Child"] = (df["Age"].fillna(100) < 13).astype(float)
    data["VIP"] = df["VIP"].fillna(False).astype(float)

    # interactions
    data["EuroChild"] = data["HomePlanet_Europa"] * data["Child"]
    data["MarsChild"] = data["HomePlanet_Mars"] * data["Child"]
    for col in ohe_cols:
        if any(s in col for s in ("Destination", "Deck")):
            data[f"Cryo_{col}"] = data["CryoSleep"] * data[col]

    # bills
    bill_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in bill_cols:
        vals = df[col].fillna(0)
        data[col] = vals / 1000.
        data[f"log{col}"] = np.log(vals + 1.)
    x = df.loc[:, bill_cols].fillna(0).to_numpy()
    log_x = np.log(x + 1)
    if not is_test:
        pca.fit(log_x)
    data["logBillsPCA"] = pca.transform(log_x)

    # group features
    data["InGroup"] = df["InGroup"].astype(float)

    return data


def read_processed_data():
    df_X = pd.read_csv("data/processed/X.csv", index_col=0)
    df_y = pd.read_csv("data/processed/labels.csv", index_col=0)
    assert (df_X.index == df_y.index).all()
    X = df_X.to_numpy()
    y = df_y.to_numpy().ravel()
    return X, df_X.columns, y


@cli.command()
def validation_curve():
    C_range = np.logspace(-2.5, 2, 20)
    X, _, y = read_processed_data()
    scores = model_selection.validation_curve(
        LogisticRegression(penalty="l1", solver="liblinear"),
        X=X,
        y=y,
        param_name="C",
        param_range=C_range,
        cv=10,
        n_jobs=-1
    )

    plt.figure()
    colors = iter("bg")
    for s, subset in zip(scores, ("train", "test")):
        mu = s.mean(1)
        sigma = s.std(1)
        c = next(colors)
        plt.semilogx(C_range, mu, color=c, marker="o", label=subset)
        plt.fill_between(C_range, mu - sigma, mu + sigma, color=c, alpha=0.15)
    plt.xlabel(r"$C$")
    plt.ylabel(r"Accuracy")
    plt.show()


@cli.command()
def learning_curve():
    X, _, y = read_processed_data()
    lr = LogisticRegression(penalty="l1", C=0.7, solver="liblinear")
    trn_sizes, trn_scores, tst_scores = model_selection.learning_curve(
        lr,
        X,
        y,
        train_sizes=np.linspace(0.1, 1.0, 12),
        cv=10,
        n_jobs=-1
    )
    plt.figure()
    for scores, c in zip([trn_scores, tst_scores], 'bg'):
        mu = scores.mean(1)
        sigma = scores.std(1)
        plt.plot(trn_sizes, mu, color=c, marker='o')
        plt.fill_between(trn_sizes, mu - sigma, mu + sigma, color=c,
                         alpha=0.15)
    plt.show()


@cli.command()
def train():
    X, features, y = read_processed_data()
    lr = LogisticRegression(C=0.7, penalty="l1", solver="liblinear")
    scores = model_selection.cross_val_score(
        estimator=lr,
        X=X,
        y=y,
        cv=10,
        n_jobs=-1
    )
    print(f"Accuracy: {scores.mean()} +- {scores.std()}")
    lr.fit(X, y)
    for name, coef in zip(features, lr.coef_.ravel()):
        print(f"{name}: {coef:.3f}")
    with open("models/logreg.pkl", "wb") as fout:
        pickle.dump(lr, fout)


@cli.command()
def create_submission():
    df = pd.read_csv("data/processed/X_test.csv", index_col=0)
    with open("models/logreg.pkl", "rb") as fin:
        lr = pickle.load(fin)
    predictions = lr.predict(df.values)
    df_out = pd.DataFrame(index=df.index)
    df_out["Transported"] = predictions.astype(bool)
    df_out.to_csv("submission.csv")


if __name__ == "__main__":
    cli()
