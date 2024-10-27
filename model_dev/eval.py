import argparse
from sklearn.metrics import confusion_matrix
import numpy as np
import mlflow
from datasets import ClassLabel, load_from_disk
from transformers import Trainer, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from utils.map_functions import tokenize_function
from utils.metrics import *
from utils.general import select_n_rows
from utils.labels_processing import *
from utils.map_functions import *
import shutil 

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description='Evaluate a model using MLflow run_id.')
parser.add_argument('--run_id', type=str, required=True, help='MLflow run ID to load the model.')
args = parser.parse_args()

# Load model artifacts using the provided run_id
model_path = f"./model/{args.run_id}/artifacts/text-classifier/model"
tokenizer_path = f"./model/{args.run_id}/artifacts/text-classifier/components/tokenizer"

# Copy the model and tokenizer directories if they exist
shutil.copytree(model_path, "./model/HF_model", dirs_exist_ok=True)
shutil.copytree(tokenizer_path, "./model/HF_model", dirs_exist_ok=True)

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./model/HF_model")
tokenizer = AutoTokenizer.from_pretrained("./model/HF_model")

# Load the dataset from disk and shuffle it
dataset = load_from_disk("./data/eval_dataset")

tokenized_dataset = dataset.map(lambda df: tokenize_function(df, tokenizer))
# Define the data collator for classification tasks
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

mlflow.set_experiment(experiment_name="Validation")  # Name of the experiment
mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")  # Where to save the results
with mlflow.start_run() as run :
    # Define the Trainer
    trainer = Trainer(
        model=model,
        eval_dataset=tokenized_dataset,  # Pass the dataset for evaluation
        data_collator=data_collator,
        compute_metrics=eval_compute_metrics
    )

        # Evaluate the model
    metrics = trainer.evaluate()
    print(metrics)  # Optionally print the evaluation metrics
