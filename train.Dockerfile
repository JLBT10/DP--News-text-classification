# Base image with Nvidia driver
FROM nvcr.io/nvidia/cuda:12.6.1-devel-ubuntu24.04
# Set the working directory
WORKDIR /src

# Update and upgrade system packages
RUN apt-get update && apt-get upgrade -y

# Install Python 3.8 and necessary development tools
RUN apt-get install -y python3 python3-dev python3-pip

# Copy requirements.txt to the working directory
COPY requirements.txt /src

# Install dependencies
RUN pip3 install -r requirements.txt --break-system-packages

# You can add more instructions or commands here if needed
