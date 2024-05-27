#!/bin/bash

if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
  pip install -r requirements.txt -U
fi

mlflow ui --backend-store-uri sqlite:///mlflow.db&

prefect server start&

wait

echo "All services started"