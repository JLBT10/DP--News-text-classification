# SETUP INSTRUCTION

Set up the project locally
```
https://github.com/JLBT10/NewsClassifier-BERT.git
```
# Hardware requirement on Amazon EC2
Os - Linux/Ubuntu
AMI - Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.3.0 (Ubuntu 20.04) 
EBS - at least 60 GB










# How to run the project!

Make a new folder in your local and clone the repository with this link.

```
https://github.com/JLBT10/NewsClassifier-BERT.git
```

Get in the books_project directory
```
cd NewsClassifier-BERT
```

**STARTING TRAINING DOCKER**

We trained the model on aws using one container for the model and another one for the mlflow server.


# Run docker compose file

2. Build the docker images
```
docker compose -f train.Dockerfile up -- build

```





3. Run the image in a container

```
docker run -it -v "/host/path":/app books

```

*/host/path : path is the books_project directory usually like this for a windows machine (c/users/.../books_project)<br>

4. Once you have started the container, run the train.py script like this and observe the outputs till the end, then exit the container.


```
python3 train.py > ./../project_report/results.txt

```
From that command, All outputs of the train.py will be directed to results.txt inside project_report folder.

With method 1 you have access to the container and the docker is mapped to the books_project directory on your local pc.<br>
**You can then easily check the results of training by opening the runs directory on your pc**.

### Method 2 
1. Open the Dockerfile and Uncomment the CMD["python3","src/train.py"] in line 19

2. Build the docker images

```
docker build . -t books -f src/Dockerfile

```
3. Run the image in a container by running the command below (model automatically start training 3 models decision tree, xgboost and random forest.)

```
docker run -it books 

```
# NB
The data analysis notebook is found in the project_report folder 







--------------------


# INTRODUCTION 

This project showcases expertise in leveraging generative AI models by optimizing and fine-tuning Large Language Models (LLMs) using pre-trained models from Hugging Face. Additionally, the project will be integrated into a CI/CD pipeline using GitHub Actions, ensuring continuous integration and high code quality standards.

To achieve our objectives, we have selected the [GPT-2 model](https://huggingface.co/openai-community/gpt2) for instruction tuning. This task aims to enhance the model's ability to understand and follow user-provided instructions. We will fine-tune GPT-2 using the [Alpaca dataset](https://huggingface.co/datasets/tatsu-lab/alpaca), which includes a diverse set of instruction-response pairs, to improve its performance across various tasks.

# SETUP INSTRUCTION

Set up the project locally
```
git clone https://github.com/JLBT10/gpt2-ft.git
```
- 
Hardware requirement on Amazon EC2
Os - Linux/Ubuntu
AMI - Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.3.0 (Ubuntu 20.04) 
EBS - at least 60 GB

# USER GUIDE

## Training the model
Once in the EC2 run the following commande to activate pytorch env :
```
source activate pytorch 
```

Building Docker file for training
```
cd gpt2-ft
sudo docker build -f ./docker/train.Dockerfile -t gpt-llm ./docker
```

Run Docker with gpus and starting the training
```
docker run -it --gpus all -v /home/ubuntu/gpt2-ft:/usr/src/app gpt-llm
```
## Running inferences with API

Building Docker file for inference
```
cd gpt2-ft
docker build -f ./docker/api.Dockerfile -t server ./docker
```

Launching FastAPI server (Uvicorn)
```
docker run -it --gpus all -v /home/ubuntu/gpt2-ft:/usr/src/app -p 80:8000 server
```
In order to visualize the default UI for testing the model through the API, you need to get the ublic IP of the Ec2 instance and map it to port 80/docs and run it in the web browser
For example 10.26.224.256:80/docs where 10.26.224.256 is the public IP of the ec2 instance.

