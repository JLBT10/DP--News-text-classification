# Base image with Nvidia driver
FROM nvcr.io/nvidia/driver:550-5.15.0-1065-nvidia-ubuntu22.04

# Set the working directory
WORKDIR /src

# Update and upgrade system packages
RUN apt-get update && apt-get upgrade -y

# Install Python 3.8 and necessary development tools
RUN apt-get install -y python3 python3-dev python3-pip

# Copy requirements.txt to the working directory
COPY requirements.txt /src

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# You can add more instructions or commands here if needed
