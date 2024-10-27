# NewsClassifier-BERT

This project is a BERT-based news classifier that has been trained and managed through a CI/CD pipeline. The model training process is automated with GitHub Actions, using a self-hosted runner on an EC2 instance. Upon code push, the CI/CD pipeline:

1. Builds a Docker image containing the training environment.
2. Launches the model training, tracking metrics and artifacts with MLflow.
3. Commits the trained model image and pushes it to Docker Hub.

The Docker images for both the training environment and model API are stored in Docker Hub, enabling easy deployment and inference. The instructions below guide you through setting up and using the model on EC2 or locally.

---

## Getting Started

These instructions will guide you through setting up the project locally or on an Amazon EC2 instance and running training and inference.

### Prerequisites

- **Amazon EC2 Hardware Requirements**
  - **OS**: Linux/Ubuntu
  - **AMI**: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.3.0 (Ubuntu 20.04)
  - **EBS**: Minimum of 60 GB

### Setup

To set up the project locally, clone the repository:
```bash
git clone https://github.com/JLBT10/NewsClassifier-BERT.git
cd NewsClassifier-BERT
```

### Docker Images

Two Docker images are used in this project:
1. `model_dev_1` - for model training with MLflow tracking
2. `model-app-1` - for serving the model and running inferences

These images are hosted on Docker Hub and can be pulled directly to your environment.

---

## User Guide

### Step 1: Training the Model

The training image has been built and saved on Docker Hub via the CI/CD pipeline, enabling you to run training directly.

#### On EC2 or Local Machine

1. Pull the Docker image required for training:
   ```bash
   docker pull jeanluc073/model_dev_1
   ```

2. Start the training process using the command below. The MLflow tracking server will be available on port `5000`:

   ```bash
   docker run -it -p 5000:5000 --gpus all jeanluc073/model_dev_1 sh run.sh
   ```

   The `run.sh` script launches MLflow and the training script:
   ```bash
   #!/bin/bash
   venv/bin/mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root /src/model_dev/mlruns --port 5000 --host 0.0.0.0 > mlflow.log 2>&1 &
   venv/bin/python3 train.py
   ```

3. **Monitoring MLflow**:
   - **EC2**: Access MLflow at `http://<EC2_PUBLIC_IP>:5000` in your browser.
   - **Localhost**: If running locally, visit `http://localhost:5000`.

After training completes, MLflow will display a `run_id`. **Take note of this `run_id`** as it will be needed for inference using the API.

---

### Step 2: Running Inference with the API

#### Start the Model API

1. Pull the Docker image for the model API:
   ```bash
   docker pull jeanluc073/model-app-1
   ```

2. Launch the container and run the API server, replacing `<RUN_ID>` with your specific `run_id` from the training step:
   ```bash
   docker run -it -p 8000:8000 jeanluc073/model-app-1 python3 server.py --run_id <RUN_ID>
   ```

#### Accessing the API

To use the API and interact with the model:
- **On EC2**:
  - **Welcome Page**: Visit `http://<EC2_PUBLIC_IP>:8000` in your browser.
  - **Prediction Interface**: Visit `http://<EC2_PUBLIC_IP>:8000/predict` for the Gradio interface for testing.

- **On Localhost**: If running locally:
  - **Welcome Page**: Go to `http://localhost:8000`.
  - **Prediction Interface**: Go to `http://localhost:8000/predict`.

---

## Summary of Commands

### Training

1. **Docker Image Pull**:
   ```bash
   docker pull jeanluc073/model_dev_1
   ```
   
2. **Start Training with MLflow Tracking**:
   ```bash
   docker run -it -p 5000:5000 --gpus all jeanluc073/model_dev_1 sh run.sh
   ```

3. **MLflow Access**:
   - **EC2**: `http://<EC2_PUBLIC_IP>:5000`
   - **Localhost**: `http://localhost:5000`

### Running the Model API

1. **Docker Image Pull**:
   ```bash
   docker pull jeanluc073/model-app-1
   ```

2. **Start API Server**:
   ```bash
   docker run -it -p 8000:8000 jeanluc073/model-app-1 python3 server.py --run_id <RUN_ID>
   ```

3. **API Access**:
   - **EC2**:
     - **Welcome Page**: `http://<EC2_PUBLIC_IP>:8000`
     - **Prediction Interface**: `http://<EC2_PUBLIC_IP>:8000/predict`
   - **Localhost**:
     - **Welcome Page**: `http://localhost:8000`
     - **Prediction Interface**: `http://localhost:8000/predict`

---

## Additional Information

- This project requires Docker with GPU support. Ensure that your EC2 instance has GPU support enabled and Docker configured to use GPUs.
- MLflow tracking is used to monitor and log the training process, allowing easy experimentation and tracking.