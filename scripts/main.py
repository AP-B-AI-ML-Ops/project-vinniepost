from prefect import flow

from load.collect import collect_flow
from train.train import train_model_flow

import mlflow

import time

name_of_dataset = 'andrewmvd/heart-failure-clinical-data'
data_path = '../data'

@flow
def main_flow():
    print('Start main flow')
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    today = time.strftime("%Y-%m-%d-")
    mlflow.set_experiment(f'heart_failure_{today}')
    
    data = collect_flow(name_of_dataset, data_path)
    hist = train_model_flow(data)
    
    
if __name__ == '__main__':
    main_flow()