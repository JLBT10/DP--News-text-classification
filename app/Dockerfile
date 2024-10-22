# Ubuntu Image
FROM python:3.12

# Set environment to non-interactive to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and create working directory
RUN apt-get update 
    #&& apt-get install -y python3.10 python3-pip 

# Set working directory
WORKDIR /src/app

# Copy files into the container
COPY . .

# Install Python dependencies
 RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Exposee the port that Uvicorn will run on
EXPOSE 8000

ENV MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlruns

