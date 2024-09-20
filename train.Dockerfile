FROM nvcr.io/nvidia/driver:550-5.15.0-1065-nvidia-ubuntu22.04

WORKDIR /src

#COPY requirements.txt /src
RUN apt-get update \
   && apt-get upgrade -y
RUN add-apt-repository universe 
RUN apt-get install -y python3.8-dev python3.8

    #&& apt-get install -y python3-pip \
    #&& python3.8 -m pip install --upgrade pip \
RUN pip install --no-cache-dir -r requirements.txt
