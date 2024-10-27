Certainly! I'll update the README to include details on accessing the MLflow tracking server during training on port 5000.

---

# NewsClassifier-BERT

This project is a BERT-based news classifier that trains on a custom dataset, with training tracked by MLflow. Model inference is made available via an API hosted on an EC2 instance.

## Getting Started

These instructions will guide you through setting up the project locally or on an Amazon EC2 instance, as well as how to run training and inference.

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

On your EC2 instance, pull the Docker image required for training:
```bash
docker pull jeanluc073/model_dev_1
```

Once the image is pulled, start the training process. The MLflow tracking server will be available on port `5000`:
```bash
docker run -it -p 5000:5000 --gpus all jeanluc073/model_dev_1 python3 train.py
```

This command will start the training script and run MLflow within the Docker container. You can monitor the training process by accessing MLflow:
- **EC2**: Visit `http://<EC2_PUBLIC_IP>:5000` in your browser to track metrics, parameters, and artifacts for each run.
  
After training completes, MLflow will display a `run_id`. **Take note of this `run_id`** as it will be needed for inference using the API.

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
- **On EC2**: Obtain the public IP of the EC2 instance and map it to port `8000` in your browser to access the API's welcome page.
  - Visit `http://<EC2_PUBLIC_IP>:8000` for the main interface.
  - Use `http://<EC2_PUBLIC_IP>:8000/predict` to access the Gradio interface for testing.

- **On a Local Computer**: If running locally, visit:
  - `http://localhost:8000` for the welcome page.
  - `http://localhost:8000/predict` for the prediction interface.

---

## Summary of Commands

### Training
```bash
docker pull jeanluc073/model_dev_1
docker run -it -p 5000:5000 --gpus all jeanluc073/model_dev_1 python3 train.py
```

### Running the Model API
```bash
docker pull jeanluc073/model-app-1
docker run -it -p 8000:8000 jeanluc073/model-app-1 python3 server.py --run_id <RUN_ID>
```

### Accessing MLflow and API Interfaces
- **MLflow (Training)**: `http://<EC2_PUBLIC_IP>:5000` 
- **Model API**:
  - **EC2**: `http://<EC2_PUBLIC_IP>:8000` or `http://<EC2_PUBLIC_IP>:8000/predict`
  - **Localhost**: `http://localhost:8000` or `http://localhost:8000/predict`

--- 

## Additional Information

- This project requires Docker with GPU support. Ensure that your EC2 instance has GPU support enabled and Docker configured to use GPUs.
- MLflow tracking is used to monitor and log the training process, allowing easy experimentation and tracking.