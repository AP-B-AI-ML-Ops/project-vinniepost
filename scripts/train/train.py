import os
import pickle
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from prefect import task, flow

def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@task
def train_model(X_train, y_train, X_test,y_test,X_val,y_val,params):
    with mlflow.start_run():
        RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state', 'n_jobs']
        for param in RF_PARAMS:
            params[param] = int(params[param])
        
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)
        
        # Dit moet nog uitgebroken worden maar moet uitzoeken hoe ik dat in dezelfde mlflow run zet
        val_rmse = mean_squared_error(y_val, rf.predict(X_val), squared=False)
        mlflow.log_metric("val_rmse", val_rmse)
        test_rmse = mean_squared_error(y_test, rf.predict(X_test), squared=False)
        mlflow.log_metric("test_rmse", test_rmse)
        

# @task
# def evaluate_model(rf,X_test,y_test,X_val,y_val):
#     val_rmse = mean_squared_error(y_val, rf.predict(X_val), squared=False)
#     mlflow.log_metric("val_rmse", val_rmse)
#     test_rmse = mean_squared_error(y_test, rf.predict(X_test), squared=False)
#     mlflow.log_metric("test_rmse", test_rmse)

