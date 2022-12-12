from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import argparse
import pickle
import sklearn.exceptions
import logging
import numpy as np
import pandas as pd
import os


# ========== CLI ========#

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", type=str)
args = parser.parse_args()

if __name__ == "__main__":
    if not os.path.exists("logging/"):
        os.mkdir("logging")

    if not os.path.exists("model/"):
        os.mkdir("model")

    # ========== LOGGING ========#

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler("logging/logfile.log")
    formatter = logging.Formatter("%(asctime)s :: %(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    # ================= LOADING IN DATA ==============#
    logger.info("Loading in CSV")
    insurance = pd.read_csv("insurance.csv")
    X = insurance.iloc[:, :-1].values
    y = insurance.iloc[:, -1].values
    logger.info("Loaded in CSV")
    # ====================== ENCODING CATEGORICAL DATA ================#
    logger.info("Encoding Categorical data")
    try:
        le_sex = LabelEncoder()
        le_smoker = LabelEncoder()
        ct = ColumnTransformer(
            transformers=[("encoder", OneHotEncoder(), [5])], remainder="passthrough"
        )

        le_sex.fit(X[:, 1])
        le_smoker.fit(X[:, 4])

        X[:, 1] = le_sex.transform(X[:, 1])
        X[:, 4] = le_smoker.transform(X[:, 4])
        X = np.array(ct.fit_transform(X))
        logger.info("Encoded Categorical data")
    except sklearn.exceptions as e:
        logger.exception(e)

    # ====================== TRAIN TEST SPLIT ================#
    logger.info("Attempting train Test split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    logger.info("Data is split")
    # ================= FITTING REGRESSION LINE =================#
    logger.info("Initialising model")
    regressor = LinearRegression()
    try:
        regressor.fit(X_train, y_train)
    except sklearn.exceptions as e:
        logger.exception(e)
    logger.info("Regressor fitted")

    # ================= TESTING REGRESSION LINE =================#
    logger.info("Predicting test data")
    y_pred = regressor.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"Error statistics: \n \tMSE: {mse} \n \tr2: {r2:.2%}")

    # ======== SAVING THE MODEL ======#
    path = args.file_path
    logger.info(f"Saving the model to {path}")
    filename = path
    pickle.dump(regressor, open(filename, "wb"))
    logger.info("Model successfully saved")
