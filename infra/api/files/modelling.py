from sklearn.pipeline import Pipeline
from preprocessing import process_data
import pandas as pd


def run_inference(payload: dict,
                  pipeline: Pipeline) -> float:
    """
    Takes a pre-fitted pipeline (dictvectorizer + linear regression model)
    outputs the computed trip duration in minutes.
    example payload:
        {'PULocationID': 264, 'DOLocationID': 264, 'passenger_count': 1}
    """
    # prep_features = process_data(payload)
    df = pd.DataFrame.from_dict(payload, orient='index').T
    df = process_data(df)

    heart_attack_prediction = pipeline.predict(df)
    return heart_attack_prediction
