#!/bin/bash
venv/bin/python3 train.py
venv/bin/mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root /src/model_dev/mlruns --port 5000 --host 0.0.0.0 > mlflow.log 2>&1 &
sleep 120
