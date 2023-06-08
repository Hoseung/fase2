__all__ = ['ColumnSelector', 'Reshaper', 'Featurizer']

# Cell
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

class ColumnSelector(BaseEstimator, TransformerMixin):
    """Sklearn pipeline to select a column from a dataframe"""
    def fit(self, column):
        self.column = column

    def transform(self, X):
        return X[self.column].values

    def inverse_transform(self, X):
        return X

class Reshaper(BaseEstimator, TransformerMixin):
    """Reshapes a numpy array from 1D to 2D"""
    def transform(self, X):
        return X.reshape(-1,1)

    def inverse_transform(self, X):
        return X.reshape(-1)

class Featurizer(BaseEstimator, TransformerMixin):
    """Featurizer which normalize a dataset to [-1,1]"""
    def __init__(self, categorical_columns):
        self.categorical_columns = categorical_columns

    def fit(self, df):
        pipelines = []
        for col in df.columns.values:
            steps = []
            column_selector = ColumnSelector()
            column_selector.fit(col)

            column_values = column_selector.transform(df)

            steps.append((col,column_selector))

            if col in self.categorical_columns:
                le = LabelEncoder()
                le.fit(column_values)
                column_values = le.transform(column_values)

                steps.append(("label_encoding",le))

            reshaper = Reshaper()
            column_values = reshaper.transform(column_values)
            steps.append(("reshape", reshaper))

            min_max = MinMaxScaler([0,1])
            min_max.fit(column_values)
            steps.append(("min_max", min_max))

            pipeline = Pipeline(steps)
            pipelines.append((col, pipeline))

        self.pipelines = FeatureUnion(pipelines)
        return self

    def transform(self, df):
        return self.pipelines.transform(df)