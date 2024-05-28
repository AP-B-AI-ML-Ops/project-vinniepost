import pickle

from pandas import DataFrame
from prefect import flow, task
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@task
def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


@task
def prepare_data(data) -> tuple:
    X = data.drop(["DEATH_EVENT"], axis=1)
    y = data["DEATH_EVENT"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )
    return X_train, X_test, y_train, y_test, X_val, y_val


@flow
def prep_flow(data: DataFrame, model_path: str):
    X_train, X_test, y_train, y_test, X_val, y_val = prepare_data(data)
    dump_pickle((X_train, y_train), f"{model_path}/train.pkl")
    dump_pickle((X_test, y_test), f"{model_path}/test.pkl")
    dump_pickle((X_val, y_val), f"{model_path}/val.pkl")
