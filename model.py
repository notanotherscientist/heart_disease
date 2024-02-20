import os
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from typing import Optional
from config import settings

from vectorizer import Vectorizer

np.random.seed(42)

class CustomLogRegression(LogisticRegression):
    def __init__(self):
        super().__init__()
        self.scaler_x: Optional[StandardScaler] = None
        self.accuracy = None

    def fit(self, X, y, sample_weight=None):
        self.scaler_x = StandardScaler().fit(X)
        X_scaled = self.scaler_x.transform(X)
        return super().fit(X_scaled, y, sample_weight)

    def predict(self, X):
        if self.scaler_x is None:
            raise AttributeError("scaler attribute is not found. Fit the model first.")

        X_scaled = self.scaler_x.transform(X)
        return super().predict(X_scaled)

    def save(self, path=None):
        file_name = "model.pkl"
        if path:
            file_name = os.path.join(path, file_name)
        with open(file_name, "wb") as f:
            pickle.dump(self, f)


def train_model(path=None):
    if not path:
        path = os.path.join(settings.PATH, 'heart.csv')
    df = pd.read_csv(path)
    df = df.drop_duplicates(inplace=False)
    vectorizer_obj = Vectorizer(df, settings.NUMERICAL_FEATURES, settings.CATEGORICAL_FEATURES)
    vectorized_df = vectorizer_obj.vectorize(df)
    target_df = df[settings.TARGET_PARAMETER]
    X_train, X_test, y_train, y_test = train_test_split(vectorized_df.values,
                                                        target_df.values.ravel(),
                                                        test_size=0.2)
    model = CustomLogRegression().fit(X_train, y_train)
    model.accuracy = round(accuracy_score(y_test, model.predict(X_test)), 2)
    return model, vectorizer_obj


