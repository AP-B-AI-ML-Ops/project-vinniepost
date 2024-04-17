from prefect import task, flow
import os
import time
import kaggle
import zipfile

os.environ['KAGGLE_USERNAME'] = 'vinniepost'
os.environ['KAGGLE_KEY'] = 'bdcecad4dbdb54f0dc50b83cc585f94f'
name_of_dataset = 'andrewmvd/heart-failure-clinical-data'

@task(retries=3, retry_delay_seconds=2)
def fetch_kaggle_data(dataset:str,data_path:str) -> None:
    kaggle.api.authenticate()
    try:
        kaggle.api.dataset_download_files(dataset, path=data_path, unzip=True)
    except Exception as e:
        print(e)
    return None

@task
def unzip_data(data_path) -> None:
    with zipfile.ZipFile(data_path + '/heart-failure-clinical-data.zip', 'r') as zip_ref:
        zip_ref.extractall(data_path)
    return None

@task
def file_creation_date(file_path:str) -> tuple:
    '''
    This function returns the creation date of a file and the time difference.
    
    '''
    creation_time = os.path.getctime(file_path)
    readable_time = time.ctime(creation_time)
    time_difference = time.time() - creation_time
    return readable_time, time_difference
    
@flow
def collect_flow(name_of_dataset:str, output_folder:str='../data') -> None:
    '''
    This function is a flow that collects data from Kaggle and unzips it.
    ---
    Args:
    --
        name_of_dataset (str): The name of the dataset on Kaggle.
    '''
    fetch_kaggle_data(name_of_dataset, output_folder)
    unzip_data(output_folder)
    return None