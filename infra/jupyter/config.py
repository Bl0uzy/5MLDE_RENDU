NUMERIC_COL = ["age", "trestbps", "chol", "thalach", "oldpeak"]
PARAM_GRID = {'n_estimators': [50, 100, 150],
              'max_depth': [None, 10, 20],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4]}
LOCAL_STORAGE = './local_storage'
TRAIN_DATA = 'data/train.csv'
TEST_DATA = 'data/test.csv'
INFERENCE_DATA = 'data/validation.csv'
SKEW_LIMIT = 0.80