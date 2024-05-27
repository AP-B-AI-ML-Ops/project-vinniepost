import os
import pickle
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from prefect import task, flow

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from keras.layers import Dense, BatchNormalization, Dropout, LSTM
from keras.models import Sequential
from keras.utils import to_categorical
from keras import callbacks
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score


@task
def load_pickle(filename):
    """
    Open a pickle file and return the contents.
    Args:
    --
        filename (str): The path to the pickle file.
    Returns:
    --
        The contents of the pickle file.
    """
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@task
def prepare_data(data) -> tuple:
    """
    function to prepare the data for training
    """
    X=data.drop(["DEATH_EVENT"],axis=1)
    y=data["DEATH_EVENT"]

    col_names = list(X.columns)
    standard_scaler = StandardScaler()
    X_df = standard_scaler.fit_transform(X)
    X_df = pd.DataFrame(X_df, columns=col_names)
    X_train, X_test, y_train,y_test = train_test_split(X_df,y,test_size=0.20,random_state=42)
    return X_train, X_test, y_train, y_test

@task
def initialise_model() -> tuple:
    """
    function to initialise the model
    """
    early_stopping = callbacks.EarlyStopping(
        min_delta=0.001, # minimium amount of change to count as an improvement
        patience=20, # how many epochs to wait before stopping
        restore_best_weights=True)

    # Initialising the NN
    model = Sequential()

    # layers
    model.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))
    model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    from keras.optimizers import SGD
    # Compiling the ANN
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model, early_stopping

@task
def train_model(X_train, y_train, model, early_stopping):
    """
    function to train the model
    """
    # Train the ANN
    history = model.fit(X_train, y_train, batch_size = 32, epochs = 500,callbacks=[early_stopping], validation_split=0.2)
    return history

@task
def evaluate_model(X_test, y_test, model):
    """
    function to evaluate the model
    """
    # Predicting the test set results
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)
    np.set_printoptions()

    return classification_report(y_test, y_pred)


@flow
def train_model_flow(data):
    """
    function to train the model
    """
    with mlflow.start_run():
        X_train, X_test, y_train, y_test = prepare_data(data)
        model, early_stopping = initialise_model()
        history = train_model(X_train, y_train, model, early_stopping)

        classification = evaluate_model(X_test, y_test, model)

        val_accuracy = np.mean(history.history['val_accuracy'])
        print(classification)
        print("\n%s: %.2f%%" % ('val_accuracy', val_accuracy*100))

    return history
