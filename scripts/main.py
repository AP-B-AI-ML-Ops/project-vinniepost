from prefect import flow

from load.collect import collect_flow

import mlflow

name_of_dataset = 'andrewmvd/heart-failure-clinical-data'
data_path = './data'

@flow
def main_flow():
    print('Start main flow')
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    
    collect_flow(name_of_dataset, data_path)
    
if __name__ == '__main__':
    main_flow()