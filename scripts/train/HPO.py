from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.data.experimental import load as tf_load
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner.tuners import RandomSearch
from prefect import task, flow
import mlflow
import os

hyperparameters = {
    'dropout_rate_1': [0.2, 0.3, 0.4],
    'dropout_rate_2': [0.2, 0.3, 0.4],
    'learning_rate': [1e-3, 1e-4, 1e-5]
}

def build_model(hp, input_shape):
    """Builds and compiles an AAN model based on hyperparameter choices."""
    model = Sequential([
        layers.Dense(16, activation='relu', input_shape=input_shape,kernel_initializer = 'uniform'),
        layers.Dense(8, activation='relu',kernel_initializer = 'uniform'),
        layers.Dropout(hp.Choice('dropout_rate_1', values=hyperparameters['dropout_rate'])),
        layers.Dense(4, activation='relu',kernel_initializer = 'uniform'),
        layers.Dropout(hp.Choice('dropout_rate_2', values=hyperparameters['dropout_rate'])),
        layers.Dense(1, activation='sigmoid',kernel_initializer = 'uniform')
    ])
    # layers
    
    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=hyperparameters['learning_rate'])),
                  metrics=['accuracy'])
    
    return model

@task
def perform_hyperparameter_optimization(train_ds, val_ds):
    tuner = RandomSearch(
        lambda hp: build_model(hp, input_shape=(train_ds.shape[1],)),  
        objective='val_accuracy',
        max_trials=1,  
        executions_per_trial=1,
        directory='tuner_dir',
        project_name='animal_classification_tuning',
        overwrite=True
    )

    with mlflow.start_run():
        tuner.search(train_ds, validation_data=val_ds, epochs=5, 
                     callbacks=[EarlyStopping(monitor='val_loss', patience=3)])
        # Log top trials
        best_trials = tuner.oracle.get_best_trials(num_trials=5)  
        for trial in best_trials:
            # Log hyperparameters
            mlflow.log_params(trial.hyperparameters.values)

            # Log metrics
            val_accuracy = trial.metrics.get_last_value("val_accuracy")
            print(val_accuracy)
            val_loss = trial.metrics.get_last_value("val_loss")
            print(val_loss)

            if val_accuracy is not None:
                mlflow.log_metric('val_accuracy', val_accuracy)
            if val_loss is not None:
                mlflow.log_metric('val_loss', val_loss)

@task
def random_sample(dataset, sample_size):
    return dataset.shuffle(buffer_size=1024).take(sample_size)

@flow
def hyperparameter_optimization_flow(train_ds=None, val_ds=None, train_sample_size=250, val_sample_size=50):

    mlflow.set_experiment("Hyperparameter_Optimization")
    
    # Take random samples of the datasets
    sampled_train_ds = random_sample(train_ds, train_sample_size)
    sampled_val_ds = random_sample(val_ds, val_sample_size)
    
    # Perform hyperparameter optimization
    perform_hyperparameter_optimization(sampled_train_ds, sampled_val_ds)
