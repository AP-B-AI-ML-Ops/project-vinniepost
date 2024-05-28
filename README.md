# Heart Failure Prediction Project

## Project Overview

Heart failure is a common event caused by cardiovascular diseases, and early detection is crucial to prevent severe health outcomes. This project develops a machine learning model to predict heart failure based on clinical features from medical records. By leveraging historical data, the model aims to identify patterns and risk factors that contribute to heart failure, which can support medical professionals in early diagnosis and intervention strategies.

## Problem Statement

Heart failure is a global health concern with high morbidity and mortality rates. Despite advances in medical science, early diagnosis remains challenging due to the complex interplay of symptoms and risk factors. This project addresses the problem by developing a predictive tool that can analyze clinical data and predict the likelihood of heart failure in patients. The goal is to provide a reliable health informatics tool that assists clinicians in making informed decisions regarding patient care.

## Dataset

The dataset used in this project is the "Heart Failure Clinical Records" dataset available on Kaggle. It includes several clinical features such as age, anaemia presence, creatinine phosphokinase levels, diabetes status, ejection fraction, high blood pressure presence, platelets count, serum creatinine levels, serum sodium levels, sex, smoking status, and time to event (death event). This data is used to train a machine learning model to predict mortality caused by heart failure.

## Project Components

- **Data Collection**: Scripts to automate the download and preprocessing of the dataset from Kaggle.
- **Data Analysis and Preprocessing**: Jupyter notebooks and scripts that perform exploratory data analysis and preprocessing steps to prepare the data for modeling.
- **Model Development**: Code for building and training machine learning models using Python's scikit-learn and TensorFlow libraries. This includes baseline models and advanced models with hyperparameter tuning.
- **Model Evaluation**: Scripts for evaluating model performance using various metrics and methodologies to ensure robustness and reliability.
- **Hyperparameter Optimization (HPO)**: Advanced scripts using Keras Tuner to find the optimal model settings to improve prediction accuracy.
- **Operationalization**: Integration of the model into a simulated production environment for real-time predictions.

## Technologies Used

- Python 3.8+
- Prefect for workflow management.
- MLflow for experiment tracking.
- Keras and TensorFlow for neural network-based models.
- Scikit-Learn for traditional machine learning models.
- Pandas and NumPy for data manipulation.

## Getting Started

To get started with this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/AP-B-AI-ML-Ops/project-vinniepost.git
```
Start the project in the devcontainer provided in the repo, Once you've done this make a kaggle.json file in the /home/vscode/.kaggle/kaggle.json with your credentials.
After you've done this you can start the mlflow server, prefect server and a pip install -r requirement.txt via the following command

```bash
sh start.sh
```

You can run the main flow by running the main.py found in the Scripts subfolder.

```bash
python Scripts/main.py
```

### Todo: Discribe how to setup a worker
