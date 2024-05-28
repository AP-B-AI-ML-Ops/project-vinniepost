import mlflow
from load.collect import collect_flow
from prefect import flow

from scripts.load.prep import prep_flow
from scripts.train.hpo import hpo_flow
from scripts.train.register import register_flow
from scripts.train.train import train_flow

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
REG_EXPERIMENT_NAME = "random-forest-best-models"
NAME_OF_SET = "andrewmvd/heart-failure-clinical-data"
DATA_PATH = "../data"
MODELS_PATH = "../models"


@flow
def main_flow():
    """
    Main flow of the pipeline
    """
    print("start main flow")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    data = collect_flow(NAME_OF_SET, DATA_PATH)
    prep_flow(data, MODELS_PATH)

    train_flow(MODELS_PATH)
    hpo_flow(MODELS_PATH, 5, HPO_EXPERIMENT_NAME)
    register_flow(MODELS_PATH, 5, REG_EXPERIMENT_NAME, HPO_EXPERIMENT_NAME)


if __name__ == "__main__":
    main_flow()
