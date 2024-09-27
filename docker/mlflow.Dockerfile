# Use the official MLflow image
FROM ghcr.io/mlflow/mlflow:v2.16.2

# Set the working directory
WORKDIR /mlflow

# Create directories for MLflow artifacts and the SQLite database
RUN mkdir -p /mlflow/mlruns

# Set environment variables for MLflow
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow/mlflow.db
#ENV MLFLOW_ARTIFACT_ROOT=/mlflow/mlruns/artifacts

# Expose the port MLflow runs on
EXPOSE 5000

# Command to run MLflow when the container starts
CMD ["mlflow", "server", "--backend-store-uri", "sqlite:///mlflow/mlflow.db", "--default-artifact-root", "/mlflow/mlruns/artifacts", "--host", "0.0.0.0", "--port", "5000"]
