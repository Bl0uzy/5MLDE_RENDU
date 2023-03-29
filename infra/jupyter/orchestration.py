import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split

import config
import pandas as pd
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.base import BaseEstimator

from typing import List

# import great_expectations

from prefect import task, flow
import great_expectations as gx

from great_expectations.core import ExpectationSuiteValidationResult
from great_expectations.checkpoint import Checkpoint

from great_expectations.checkpoint import LegacyCheckpoint

import mlflow
from mlflow import MlflowClient

from ruamel import yaml
from great_expectations.core.batch import RuntimeBatchRequest





@task(name="Load", tags=['Serialize'])
def load_pickle(path: str):
    with open(path, 'rb') as f:
        loaded_obj = pickle.load(f)
    return loaded_obj


@task(name="Save", tags=['Serialize'])
def save_pickle(path: str, obj: dict):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

@task(name='load_data', tags=['preprocessing'], retries=2, retry_delay_seconds=60)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@flow(name="Data validation")
def run_ge_checkpoint(
        path: str,
):
    """
    Load data from path and check it with great expectation
    raise a exception if data is not valid
    """

    context = gx.get_context()

    datasource_config = {
        "name": "example_datasource",
        "class_name": "Datasource",
        "module_name": "great_expectations.datasource",
        "execution_engine": {
            "module_name": "great_expectations.execution_engine",
            "class_name": "PandasExecutionEngine",
        },
        "data_connectors": {
            "default_runtime_data_connector_name": {
                "class_name": "RuntimeDataConnector",
                "module_name": "great_expectations.datasource.data_connector",
                "batch_identifiers": ["default_identifier_name"],
            },
        },
    }

    context.test_yaml_config(yaml.dump(datasource_config))

    context.add_datasource(**datasource_config)

    df = pd.read_csv(path)

    batch_request = RuntimeBatchRequest(
        datasource_name="example_datasource",
        data_connector_name="default_runtime_data_connector_name",
        data_asset_name="test2",  # This can be anything that identifies this data_asset for you
        runtime_parameters={"batch_data": df},  # df is your dataframe
        batch_identifiers={"default_identifier_name": "default_identifier"},
    )

    validator = context.get_validator(
        batch_request=batch_request, expectation_suite_name="heart_attack_expectations_suite"
    )

    validation_result = validator.validate()

    if (validation_result.success == False):
        raise Exception("Data didn't pass gx")

@task(name='get_x', tags=['preprocessing'])
def get_x(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Retrieve X from dataFrame
    :return df.iloc[:,['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca','thal']]
    """
    X = df.loc[:,
        ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
         'thal']]
    return X


@task(name='get_y', tags=['preprocessing'])
def get_y(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Retrieve X from dataFrame
    :return df.target
    """
    # y = df[['target']]
    y = df.target
    return y


@task(name='fix_skew', tags=['preprocessing'])
def fix_skew(
        x: pd.DataFrame,
        skew_limit : int = 0.80,
) -> pd.DataFrame:
    """
    Find skew cols from dataframe and use FunctionTransformer to normalise values
    :return df with fix skew
    """
    skew_vals = x[config.NUMERIC_COL].skew()
    skew_col = skew_vals[abs(skew_vals) > skew_limit].sort_values(ascending=False)
    ft = FunctionTransformer(func=np.log1p)
    skew = skew_col.index.tolist()
    x[skew] = ft.fit_transform(x[skew])

    return x


@task(name='normalize_data', tags=['preprocessing'])
def normalize_data(
        x: pd.DataFrame,
) -> pd.DataFrame:
    """
    Use StandardScaler on numerical cols (all cols for this df)
    """
    standardscaler = StandardScaler()
    standardscaler.fit(x)
    x = pd.DataFrame(standardscaler.transform(x), columns=x.columns[:])
    # x = standardscaler.transform(x)

    return x


@flow(name="Data processing", retries=1, retry_delay_seconds=30)
def process_data(path: str, with_target: bool = True, skew_limit: int = 0.80) -> dict:
    """
    Load data from a csv
    Normalize data and apply fix skew (optional)
    :return final X and y
    """
    df = load_data(path)



    X = get_x(df)
    X = fix_skew(X, skew_limit)
    X = normalize_data(X)
    y = None
    if with_target:
        y = get_y(df)

    return {'X': X, 'y': y}


@task(name="Get best hyperparamters", tags=['Model'])
def find_best_hyperparameters(X: pd.DataFrame, y: pd.DataFrame, param_grid: dict = config.PARAM_GRID) -> dict:
    """"
    Start a gridSearchCV
    :return dict for best hyperparameters
    """

    # Create a GridSearchCV object
    grid_search = GridSearchCV(estimator=RandomForestClassifier(),
                               param_grid=param_grid,
                               cv=5,
                               n_jobs=-1)

    grid_search.fit(X, y)

    return grid_search.best_params_


@task(name="Train model", tags=['Model'])
def train_model(
        x_train: np.ndarray,
        y_train: np.ndarray,
        hyperparameters: dict
) -> RandomForestClassifier:
    """Train and return a RandomForestClassifier model"""
    model = RandomForestClassifier(**hyperparameters)
    model.fit(x_train, y_train)
    return model


@task(name="Make prediction", tags=["Model"])
def predict_target(
        input_data: np.ndarray,
        model: RandomForestClassifier
) -> np.ndarray:
    """
    Use trained RandomForestClassifier model
    to predict target from input data
    :return array of predictions
    """
    return model.predict(input_data)


@task(name="Evaluation", tags=["Model"])
def evaluate_model(
        y_test: np.ndarray,
        y_pred: np.ndarray
) -> float:
    """Calculate accuracy"""
    return accuracy_score(y_test, y_pred)


@flow(name="Model initialisation")
def train_and_predict(
        x_train,
        y_train,
        x_test,
        y_test
) -> dict:
    """Train model, predict values and calculate error"""
    hyperparams = find_best_hyperparameters(x_train, y_train)
    model = train_model(x_train, y_train, hyperparams)
    prediction = predict_target(x_test, model)
    accuracy = evaluate_model(y_test, prediction)
    return {'model': model, 'accuracy': accuracy, 'hyperparameters':hyperparams}


@flow(name="Machine learning workflow", retries=1, retry_delay_seconds=30)
def complete_ml(
        train_path: str,
        test_path: str,
        save_model: bool = True,
        local_storage: str = config.LOCAL_STORAGE
) -> None:
    """
    Load data and prepare data for model training
    Search best hyperparameters, train model, make predictions and calculate accuracy
    Save model in mlflow and set tit to prod if better
    :return none
    """

    if not os.path.exists(local_storage):
        os.makedirs(local_storage)

    # train_data = process_data(train_path)
    # test_data = process_data(test_path)

    mlflow.set_tracking_uri('http://mlflow:5000')
    mlflow_experiment_path = f"/mlflow/hearth_attack_predi"
    mlflow.set_experiment(mlflow_experiment_path)

    metric_name = 'Accuracy'

    client = MlflowClient()
    experiment = dict(client.get_experiment_by_name(mlflow_experiment_path))
    runs_list = client.search_runs([experiment['experiment_id']])

    best_metric = 0

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tag("Level", "Development")
        mlflow.set_tag("Team", "Data Science")
        mlflow.log_param("skew_limit", config.SKEW_LIMIT)
        train_data = process_data(train_path,True,config.SKEW_LIMIT)
        test_data = process_data(test_path,True,config.SKEW_LIMIT)
        mlflow.log_param("train_set_size", train_data['X'].shape[0])
        mlflow.log_param("test_set_size", test_data['X'].shape[0])
        model_obj = train_and_predict(train_data['X'], train_data['y'], test_data['X'],test_data['y'])
        mlflow.log_param("HyperParameters", model_obj['hyperparameters'])
        mlflow.log_metric("Accuracy", model_obj['accuracy'])
        mlflow.sklearn.log_model(model_obj['model'], "models")

        print(best_metric, ' < ', model_obj['accuracy'])

        for run in runs_list:
            # retrieve run info
            run_dict = run.to_dictionary()
            single_run_id = run_dict['info']['run_id']
            if (single_run_id != run_id):
                # extract the metrics
                metrics = run_dict['data']['metrics']
                # retrieve historical metrics
                print(metrics)
                # print(best_metric, ' < ', metrics['Accuracy'])
                if 'accuracy' in metrics and best_metric < metrics['Accuracy']):
                    best_metric = metrics['Accuracy']

        if 'accuracy' in model_obj and best_metric < model_obj['accuracy']:
            print('Pass to prod')
            production_version = mlflow.register_model(f"runs:/{run_id}/models", "hearth_attack_predi")
            # production_version = 1
            print(production_version)
            client.transition_model_version_stage(
                name="hearth_attack_predi", version=production_version.version, stage="Production"
            )

        if save_model:
            save_pickle(f"{local_storage}/model.pickle", model_obj)


@flow(name="Batch inference", retries=1, retry_delay_seconds=30)
def batch_inference(input_path, model=None, local_storage=config.LOCAL_STORAGE):
    """
    Load model from folder
    Transforms input data
    Predict values using loaded model
    :return array of predictions
    """
    data = process_data(input_path, False)
    if not model:
        model = load_pickle(f"{local_storage}/model.pickle")["model"]
    return predict_target(data["X"], model)


# data = process_data('hearth_attack.csv')
# find_best_hyperparameters(data['X'], data['y'])
# complete_ml('hearth_attack.csv', True)
if __name__ == "__main__":
    complete_ml(config.TRAIN_DATA,config.TEST_DATA, True)
# run_ge_checkpoint(config.TRAIN_DATA)
# inference = batch_inference(config.INFERENCE_DATA)
# print(inference)