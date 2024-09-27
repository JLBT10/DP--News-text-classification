import mlflow
#from transformers import AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import torch
from mlflow.models.signature import infer_signature
from mlflow import MlflowClient


# On donne un nom à l'expérience
mlflow.set_experiment("classification of news")

# On donne un nom à l'expérience
#experiment =  
mlflow.set_experiment(experiment_name="boou")
#experiment_id = mlflow.create_experiment(name="test")
# On commence l'experience
# On se connecte à l'interface UI et la base de donnée
mlflow.set_tracking_uri("http://mlflow:5000")
#mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db") 
with mlflow.start_run(run_name="Logging params"):#,experiment_id=experiment.experiment_id) as runs:
    parameters = {'learning_rate':0.1,
    "loss":"mse",
    "optimizer":"Adam"}
    metrics = {"mse":0.7,
    "mae":0.3}
    mlflow.log_metrics(metrics)
    mlflow.log_params(parameters)

    with open("test.txt","w") as f:
        f.write("hello the world")

    model = torch.nn.Linear(10, 2)  # A simple linear layer model
    fig_roc = plt.figure()

    mlflow.log_artifact(local_path="test.txt",artifact_path="Text_files")
    mlflow.log_figure(fig_roc,"metrics/auc_metric.png")
    mlflow.pytorch.log_model(model,artifact_path='model')
