# Ubuntu Image
FROM ubuntu:22.04

# Set environment to non-interactive to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and create working directory
RUN apt-get update \
    && apt-get install -y python3.10 python3-pip \
    && mkdir -p /src

# Set working directory
WORKDIR /src

# Copy files into the container
COPY requirements.txt requirements.txt
COPY . /src/app

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Exposee the port that Uvicorn will run on
EXPOSE 8000
