import time

import mlflow
from load.collect import collect_flow
from prefect import flow
from train.HPO import hyperparameter_optimization_flow
from train.train import train_model_flow

NAME_OF_SET = "andrewmvd/heart-failure-clinical-data"
DATA_PATH = "../data"


@flow
def main_flow():
    print("Start main flow")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    today = time.strftime("%Y-%m-%d-")
    mlflow.set_experiment(f"heart_failure_{today}")
    data = collect_flow(NAME_OF_SET, DATA_PATH)
    model = train_model_flow(data)
    best_model = hyperparameter_optimization_flow(data, data)
    print("End main flow")
    # Save model and best_model in mlflow
    mlflow.keras.log_model(model, "model")
    mlflow.keras.log_model(best_model, "best_model")
    mlflow.end_run()


if __name__ == "__main__":
    main_flow()
