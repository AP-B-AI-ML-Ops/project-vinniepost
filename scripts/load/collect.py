"""
Script containing the flow for collecting data from Kaggle.
"""
import logging
import os
import time

import kaggle
from prefect import flow, task

import pandas as pd

os.environ['KAGGLE_USERNAME'] = 'vinniepost'
os.environ['KAGGLE_KEY'] = 'bdcecad4dbdb54f0dc50b83cc585f94f'
NAME_OF_DATASET = 'andrewmvd/heart-failure-clinical-data'

@task(retries=3, retry_delay_seconds=2)
def fetch_kaggle_data(dataset:str,data_path:str) -> None:
    """
    Function to fetch data from Kaggle. 
    This task will try to download 3 times with a 2 second delay between each try.
    ---
    Args:
    --
        dataset (str): The name of the dataset on Kaggle.
        data_path (str): The path to save the data.
    Returns:
    --
        None
    """
    kaggle.api.authenticate()
    try:
        kaggle.api.dataset_download_files(dataset, path=data_path, unzip=True)
    except Exception as error:
        logging.error(error)
        print(error)

@task
def pass_data() -> pd.DataFrame:
    """
    passes the data in a dataframe to the next task
    """
    data = pd.read_csv("../data/heart_failure_clinical_records_dataset.csv")
    return data

@flow
def collect_flow(name_of_dataset:str, output_folder:str='../data') -> pd.DataFrame:
    '''
    This function is a flow that collects data from Kaggle and unzips it.
    ---
    Args:
    --
        name_of_dataset (str): The name of the dataset on Kaggle.
    '''
    fetch_kaggle_data(name_of_dataset, output_folder)
    data = pass_data()
    return data
