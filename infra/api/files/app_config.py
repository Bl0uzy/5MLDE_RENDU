import pandas as pd

# MLFLOW
MLFLOW_TRACKING_URI = "http://mlflow:5000"

STAGE = "Production"
REGISTERED_MODEL_NAME = "hearth_attack_predi"
REGISTERED_MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}/{STAGE}"

# MISC
APP_TITLE = "HeartAttackPrediction"
APP_DESCRIPTION = ("A simple API to predict heart attack. Using a model pulled from MLFlow.")
APP_VERSION = "0.0.1"
# silence pandas `SettingWithCopyWarning` warnings
pd.options.mode.chained_assignment = None  # default='warn'