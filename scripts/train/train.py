import logging

import mlflow
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential
from prefect import flow, task
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)


@task
def prepare_data(data) -> tuple:
    X = data.drop(["DEATH_EVENT"], axis=1)
    y = data["DEATH_EVENT"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42
    )
    return X_train, X_test, y_train, y_test


@task
def initialise_model(input_dim: int) -> Sequential:
    model = Sequential(
        [
            Dense(16, activation="relu", input_dim=input_dim),
            Dense(8, activation="relu"),
            Dropout(0.25),
            Dense(4, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


@task
def train_model(X_train, y_train, model):
    early_stopping = EarlyStopping(
        min_delta=0.001, patience=20, restore_best_weights=True
    )
    history = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=500,
        callbacks=[early_stopping],
        validation_split=0.2,
    )
    return history


@flow
def train_model_flow(data):
    with mlflow.start_run():
        X_train, _, y_train, _ = prepare_data(data)
        model = initialise_model(X_train.shape[1])
        history = train_model(X_train, y_train, model)
        logging.info(
            f"Validation Accuracy: {np.mean(history.history['val_accuracy']) * 100:.2f}%"
        )
    return model
