# Base image with NVIDIA CUDA and development drivers
FROM nvcr.io/nvidia/cuda:12.6.1-devel-ubuntu24.04

# Set the working directory
WORKDIR /src

# Update system packages and install necessary dependencies for Python development
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y python3 python3-dev python3-pip python3-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a virtual environment for Python package management
RUN python3 -m venv venv

# Upgrade pip within the virtual environment to the latest version
RUN ./venv/bin/pip install --upgrade pip

COPY requirements.txt /src

# Install Python dependencies from requirements.txt within the virtual environment
RUN ./venv/bin/pip install --no-cache-dir -r requirements.txt

# Set the PATH environment variable to include the virtual environment's bin directory
ENV PATH="/src/venv/bin:$PATH"

ENV GIT_PYTHON_REFRESH=quiet

# Default command to run the training script
CMD ["python3","model_dev/eval.py"]
