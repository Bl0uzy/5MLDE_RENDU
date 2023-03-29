from typing import List

import pandas as pd
import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.base import BaseEstimator


# def load_data(path: str) -> pd.DataFrame:
#     return pd.read_csv(path)

def get_x(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Split original data in X and Y
    :return {'x': df.iloc[:,0:-1], 'y': df.target}
    """
    X = df.loc[:,
        ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
         'thal']]
    return X


def normalize_data(
        x: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find skew cols from dataframe and use FunctionTransformer to normalise values
    """
    standardscaler = StandardScaler()
    standardscaler.fit(x)
    x = pd.DataFrame(standardscaler.transform(x), columns=x.columns[:])

    return x


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load data from a csv
    Compute target (duration column) and apply threshold filters (optional)
    Turn features to sparce matrix
    :return final X and y
    """

    X = get_x(df)
    X = normalize_data(X)

    return X

