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
COPY requirements.txt /src/requirements.txt
COPY . /src/app

# Install Python dependencies
RUN pip3 install --upgrade pip \
    && pip3 install -r requirements.txt \
    && pip3 install uvicorn[standard]
RUN pip3 install datasets==3.0.0
RUN pip install --upgrade datasets
RUN pip install awscli --break-system-packages \
    && pip install dvc-s3 --break-system-packages 
   # && aws s3 cp s3://your-bucket/path/to/model.tar.gz ./model.tar.gz


# Expose the port that Uvicorn will run on
EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
#uvicorn app.server:app --host 0.0.0.0 --port 8000