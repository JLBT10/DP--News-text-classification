# SETUP INSTRUCTION

Set up the project locally
```
https://github.com/JLBT10/NewsClassifier-BERT.git
```
# Hardware requirement on Amazon EC2
Os - Linux/Ubuntu
AMI - Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.3.0 (Ubuntu 20.04) 
EBS - at least 60 GB

# USER GUIDE

## Training the model
Once in the EC2 run the following commande to activate pytorch env :
```
source activate pytorch 
```

Launch docker compose for building of the mflow and training dockerfile 
```
cd NewsClassifier-BERT
docker compose -f docker-compose-train.yaml up --build 
```

## Running inferences with API

To launch mlflow container server and the app sever : 
```
cd NewsClassifier-BERT
docker compose -f docker-compose-app.yaml up --build 
```

In order to visualize the default UI for testing the model through the API, you need to get the ublic IP of the Ec2 instance and map it to port 80/docs and run it in the web browser. 
For example 10.26.224.256:80/docs where 10.26.224.256 is the public IP of the ec2 instance.