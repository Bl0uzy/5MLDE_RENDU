import pandas as pd
import os
import mlflow
import pickle

def get_x(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Split original data
    :return DataFrame without intrusive columns
    """
    X = df.loc[:,
        ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
         'thal']]
    return X


def normalize_data(
        x: pd.DataFrame,
        run_id: str
) -> pd.DataFrame:
    """
    Retrieve the scaler from mlflow using pickle
    Transform data
    """

    client = mlflow.tracking.MlflowClient()
    local_dir = "/tmp/artifact_downloads"
    if not os.path.exists(local_dir):
        os.mkdir(local_dir)
    client.download_artifacts(run_id, '', local_dir)

    with open('/tmp/artifact_downloads/ss.pickle', 'rb') as f:
        ss = pickle.load(f)
    transformed_data = ss.transform(x)
    x = pd.DataFrame(transformed_data, columns=x.columns[:])

    return x


def process_data(
        df: pd.DataFrame,
        run_id: str
) -> pd.DataFrame:
    """
    Format the dataset
    Normalise values
    :return final X
    """

    X = get_x(df)
    X = normalize_data(X, run_id)

    return X

