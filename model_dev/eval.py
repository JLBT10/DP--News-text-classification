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

# Load the model and tokenizer (you can replace with your specific model)
model_name = "model_dev/runs/best_model"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the dataset from disk and shuffle it
dataset = load_from_disk("./model_dev/data/inshort_dataset").shuffle(seed=42).select(range(2))

### Processing Labels
labels = dataset.unique("labels") # Get a list of unique labels
label2id, id2label = get_label2id_id2label(labels) # Get the mapping label2id and id2label


features_class = ClassLabel(names=labels) #Define ClassLabel in order to stratified split wr to labels columns

dataset = dataset.map(lambda example : convert_label_to_id(example,label2id))
dataset = dataset.cast_column("labels", features_class) # Insert it into the dataset columns

tokenized_dataset = dataset.map(lambda df: tokenize_function(df,tokenizer))
 ### Let's define the data collator for classification tasks
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
mlflow.set_experiment(experiment_name="Validation") # Name of the experience
mlflow.set_tracking_uri("sqlite:///model_dev/mlruns/mlflow.db") # Where to save the result

# Define the Trainer
trainer = Trainer(
    model=model,
    eval_dataset=tokenized_dataset,  # Pass the dataset for evaluation
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Evaluate the model
metrics = trainer.evaluate()

print(metrics)
