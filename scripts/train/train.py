import os
import pickle

import mlflow
from prefect import flow, task
from sklearn.ensemble import RandomForestRegressor


@task
def load_pickle(filename: str):
    """Helper task to load a pickle file"""
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@task
def start_ml_experiment(X_train, y_train):
    """Task that starts the mlflow experiment"""
    with mlflow.start_run():
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)


@flow
def train_flow(model_path: str):
    """The train flow"""
    mlflow.set_experiment("random-forest-train")
    mlflow.sklearn.autolog()
    X_train, y_train = load_pickle(os.path.join(model_path, "train.pkl"))
    X_train = X_train.copy()
    y_train = y_train.copy()

    start_ml_experiment(X_train, y_train)
