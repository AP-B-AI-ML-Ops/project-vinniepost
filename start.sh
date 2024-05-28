#!/bin/bash

if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
fi

mlflow ui --backend-store-uri sqlite:///scripts/mlflow.db&

prefect server start&

wait

echo "All services started"
