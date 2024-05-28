import mlflow
from keras_tuner.tuners import RandomSearch
from prefect import flow, task
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

hyperparameters = {
    "dropout_rate_1": [0.2, 0.3, 0.4],
    "dropout_rate_2": [0.2, 0.3, 0.4],
    "learning_rate": [1e-3, 1e-4, 1e-5],
}


def build_model(hp):
    """Builds and compiles an ANN model based on hyperparameter choices."""
    model = Sequential(
        [
            layers.Dense(
                16, activation="relu", input_shape=(12,), kernel_initializer="uniform"
            ),
            layers.Dense(8, activation="relu", kernel_initializer="uniform"),
            layers.Dropout(
                hp.Choice("dropout_rate_1", values=hyperparameters["dropout_rate_1"])
            ),
            layers.Dense(4, activation="relu", kernel_initializer="uniform"),
            layers.Dropout(
                hp.Choice("dropout_rate_2", values=hyperparameters["dropout_rate_2"])
            ),
            layers.Dense(1, activation="sigmoid", kernel_initializer="uniform"),
        ]
    )
    model.compile(
        optimizer=Adam(
            hp.Choice("learning_rate", values=hyperparameters["learning_rate"])
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


@task
def perform_hyperparameter_optimization(train_ds, val_ds):
    tuner = RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=10,  # More appropriate number of trials
        executions_per_trial=3,  # More trials for each configuration
        directory="tuner_dir",
        project_name="hyperparam_opt",
        overwrite=True,
    )
    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5)],
    )
    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model


@flow
def hyperparameter_optimization_flow(train_ds, val_ds) -> Sequential:
    mlflow.set_experiment("Hyperparameter_Optimization")
    best_model = perform_hyperparameter_optimization(train_ds, val_ds)
    mlflow.log_artifacts("tuner_dir")  # Log the directory for future reference
    return best_model


if __name__ == "__main__":
    # Example datasets
    train_ds = None  # Replace with actual TensorFlow dataset
    val_ds = None  # Replace with actual TensorFlow dataset
    hyperparameter_optimization_flow(train_ds, val_ds)
