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
  - **Operating System**: Linux/Ubuntu
  - **AMI**: Deep Learning AMI with NVIDIA Driver & PyTorch 2.3.0 (Ubuntu 20.04)
  - **Storage**: Minimum of 100 GB EBS volume

#### EC2 Instance Configuration Steps

1. **When creating the EC2 instance, ensure the following options are selected:**
   - **Allow SSH traffic from your IP**
   - **Allow HTTP and HTTPS traffic from the internet**

2. **Update Security Group to Open Required Ports:**
   - Go to the EC2 Security Group settings.
   - Add two custom TCP rules:
     - **Port 5000**: To access MLflow's tracking server.
     - **Port 8000**: To test and interact with the model’s API interface.
     - 
This setup enables secure access for SSH and web traffic, along with open ports to view MLflow logs and test the model from your browser.

### Setup

To set up the project locally, clone the repository:
```bash
git clone https://github.com/JLBT10/NewsClassifier-BERT.git
cd NewsClassifier-BERT
```

### Docker Images

Two Docker images are used in this project:
1. `model_dev_2` - for model training with MLflow tracking
2. `model-app-2` - for serving the model and running inferences

These images are hosted on Docker Hub and can be pulled directly to your environment.

---

## User Guide
### NB
If you do not wish to train on your machine, you could test the already trained and saved model by running these two commands.

1. Pull the Docker image for the model API:
   ```bash
   docker pull jeanluc073/model-app-2
   ```

2. Launch the container and start the API server, using `d8db502f9ce541289052a3ca85c4877b` as the argument for the run_id obtained during the training step:
   ```bash
   docker run -it -p 8000:8000 jeanluc073/model-app-2 python3 server.py --run_id d8db502f9ce541289052a3ca85c4877b
   ```


### Step 1: Training the Model

The training image has been built and saved on Docker Hub via the CI/CD pipeline, enabling you to run training directly.

#### On EC2 or Local Machine

1. Pull the Docker image required for training:
   ```bash
   docker pull jeanluc073/model_dev_2
   ```

2. Start the training process using the command below. The MLflow tracking server will be available on port `5000`:

   ```bash
   mkdir -p mlruns
   docker run -it -p 5000:5000 --gpus all -v ./mlruns:/src/model_dev/mlruns jeanluc073/model_dev_2 sh run.sh
   ```
  The `run.sh` script launches MLflow and the training script.


3. **Monitoring MLflow**:
   - **EC2**: Access MLflow at `http://<EC2_PUBLIC_IP>:5000` in your browser.
   - **Localhost**: If running locally, visit `http://localhost:5000`.

After training completes, MLflow will log a unique `run_id` associated with this specific model training session. **Make sure to save this `run_id`, as it will be required to perform inferences via the API.** 

For convenience, in this setup, the `run_id` of `d8db502f9ce541289052a3ca85c4877b` corresponds to the primary training session for this model and already available in the container if you wish to try it. This `run_id` allows the API container to access the correct model artifacts, ensuring the model used for inference matches the trained configuration and parameters.

---

### Step 2: Running Inference with the API

#### Start the Model API

1. Pull the Docker image for the model API:
   ```bash
   docker pull jeanluc073/model-app-2
   ```

2. Launch the container with the run_id that you got from your training. Use it to run the server.py :
   ```bash
   docker run -it -p 8000:8000 jeanluc073/model-app-2 python3 server.py --run_id <RUN_ID>
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

## Additional Information

- This project requires Docker with GPU support. Ensure that your EC2 instance has GPU support enabled and Docker configured to use GPUs.
- MLflow tracking is used to monitor and log the training process, allowing easy experimentation and tracking.
