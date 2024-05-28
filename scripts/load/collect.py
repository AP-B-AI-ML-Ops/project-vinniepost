import logging
import os

import kaggle
import pandas as pd
from prefect import flow, task

# Set up logging
logging.basicConfig(level=logging.INFO)


@task(retries=3, retry_delay_seconds=2)
def fetch_kaggle_data(dataset: str, data_path: str):
    kaggle.api.authenticate()
    try:
        kaggle.api.dataset_download_files(dataset, path=data_path, unzip=True)
    except Exception as error:
        logging.error(f"Failed to download dataset {dataset}: {str(error)}")


@task
def pass_data(data_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(
            os.path.join(data_path, "heart_failure_clinical_records_dataset.csv")
        )
        return data
    except Exception as error:
        logging.error(f"Failed to read data from {data_path}: {str(error)}")
        raise


@flow
def collect_flow(
    name_of_dataset: str = "andrewmvd/heart-failure-clinical-data",
    output_folder: str = "../data",
) -> pd.DataFrame:
    fetch_kaggle_data(name_of_dataset, output_folder)
    return pass_data(output_folder)


if __name__ == "__main__":
    collect_flow("andrewmvd/heart-failure-clinical-data")
