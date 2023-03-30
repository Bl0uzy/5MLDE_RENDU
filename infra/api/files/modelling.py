from sklearn.pipeline import Pipeline
from preprocessing import process_data
import pandas as pd


def run_inference(payload: dict,
                  pipeline: Pipeline) -> float:
    """
    Takes a pre-fitted pipeline (dictvectorizer + linear regression model)
    outputs if there is risk of heart attack (boolean).
    example payload:
        {"age": 37,"sex": 1,"cp": 2,"trestbps": 130,"chol": 250,"fbs": 0,"restecg": 1,"thalach": 187,"exang": 0,"oldpeak": 3.5,"slope": 0,"ca": 0,"thal": 2}
    """
    df = pd.DataFrame.from_dict(payload, orient='index').T
    print(pipeline._model_meta)
    df = process_data(df, pipeline._model_meta.run_id)

    heart_attack_prediction = pipeline.predict(df)
    return heart_attack_prediction
