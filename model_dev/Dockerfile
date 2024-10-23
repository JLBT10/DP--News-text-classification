# Base image with NVIDIA CUDA and development drivers
FROM nvcr.io/nvidia/cuda:12.6.1-devel-ubuntu24.04

# Add a path the the path env

ENV GIT_PYTHON_REFRESH=quiet
# Set the working directory
WORKDIR /src/model_dev
ENV PATH="/src/model_env/venv/bin:$PATH"

# Update system packages and install necessary dependencies for Python development
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y python3 python3-dev python3-pip python3-venv \
    && apt-get clean \ 
    && rm -rf /var/lib/apt/lists/*

# Copy all the file in the model_dev to the docker image
COPY . .

# Create a Python virtual environment
RUN python3 -m venv venv \
    && venv/bin/pip install --upgrade pip \
    && venv/bin/pip install --no-cache-dir -r requirements.txt

# Set the PATH environment variable to include the virtual environment's bin directory

# Create the artifacts directory for MLflow if it doesn't exist
RUN mkdir -p /src/model_dev/mlruns/artifacts

ENV MLFLOW_TRACKING_URI=sqlite:////src/model_dev/mlruns/mlflow.db
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=/src/model_dev/mlruns/artifacts

EXPOSE 5000

