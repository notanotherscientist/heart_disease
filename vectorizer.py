import pandas as pd
import os
import pickle


class Vectorizer():

    def __init__(self, initial_df, numerical_features, categorical_features):
        self.numerical_cols = numerical_features
        self.categorical_features = categorical_features
        self.categorical_cols = pd.get_dummies(initial_df[categorical_features].astype(str)).columns.tolist()
        self.x_vector_cols = self.categorical_cols + self.numerical_cols

    def vectorize(self, df):
        categorical_intersection = list(set(self.categorical_features) & set(df.columns))
        df[categorical_intersection] = df[categorical_intersection].astype(str)
        df = pd.get_dummies(df)
        df = df.reindex(columns=self.x_vector_cols, fill_value=0)
        return df

    def save(self, path=None):
        file_name = "vectorizer.pkl"
        if path:
            file_name = os.path.join(path, file_name)
        with open(file_name, "wb") as f:
            pickle.dump(self, f)
