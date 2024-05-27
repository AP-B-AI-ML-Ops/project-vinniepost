from prefect import flow

from load.collect import collect_flow
from train.train import train_model_flow

import mlflow

import time

NAME_OF_SET = 'andrewmvd/heart-failure-clinical-data'
DATA_PATH = '../data'

@flow
def main_flow():
    print('Start main flow')
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    today = time.strftime("%Y-%m-%d-")
    mlflow.set_experiment(f'heart_failure_{today}')
    
    data = collect_flow(NAME_OF_SET, DATA_PATH)
    model = train_model_flow(data)
    
    
if __name__ == '__main__':
    main_flow()