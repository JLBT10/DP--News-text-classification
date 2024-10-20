# Launch mlflow server
FROM  ghcr.io/mlflow/mlflow:v2.16.2

services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.16.2 # Pre-built MLflow image
    ports:
      - "5001:5000"  # Expose port 5000 in the container to 5001 on the host
    volumes:
      - ./mlruns:/mlruns  # Mount local folder for MLflow artifacts
    environment:
      #- MLFLOW_TRACKING_URI=sqlite:///mlflow/mlflow.db  # Using SQLite as backend for tracking
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlruns/artifacts  # Path for artifact storage
    command:
      mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --host 0.0.0.0 --port 5000 
