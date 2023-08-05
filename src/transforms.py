from sklearn.base import BaseEstimator, TransformerMixin


class InteractionTransformer(BaseEstimator, TransformerMixin):
    "Modifies input dataframe"
    def __init__(self, pairs: list[tuple[str, str]]):
        super().__init__()
        self.pairs = pairs

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for col1, col2 in self.pairs:
            X[f"{col1}_{col2}"] = X[col1] * X[col2]
        return X
