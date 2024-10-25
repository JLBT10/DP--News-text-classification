""" Training of the model """

### Import necessary libraries
import mlflow
import os
# Transformers Library Imports
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer, pipeline
)
from mlflow.models import infer_signature
from datasets import ClassLabel, load_from_disk

# Project-Specific imports
from utils.metrics import compute_metrics
from utils.general import select_n_rows
from utils.labels_processing import *
from utils.map_functions import *

if __name__ == '__main__':
    ### Defining constant
    DATASET_PATH = './data/inshort_dataset' #Dataset path on github
    CHECKPOINT = "bert-base-cased" # Name of the model
    

    ### Loading data
    inshort_data = load_from_disk(DATASET_PATH)

    ### Processing Labels
    labels = inshort_data.unique("labels") # Get a list of unique labels
    label2id, id2label = get_label2id_id2label(labels) # Get the mapping label2id and id2label

    
    features_class = ClassLabel(names=labels) #Define ClassLabel in order to stratified split wr to labels columns
    
    inshort_data = inshort_data.map(lambda example : convert_label_to_id(example,label2id))
    #convert_label_to_id(inshort_data,label2id)
    inshort_data = inshort_data.cast_column("labels", features_class) # Insert it into the dataset columns
   
    #features_class
    inshort_data = inshort_data.cast_column("labels", features_class) # Insert it into the dataset columns

    ### Split the dataset into train and validation using labels columns to stratify (it handles imbalance)
    inshort_data = inshort_data.train_test_split(test_size=0.3,shuffle=True, stratify_by_column="labels")
    
    ### Processing data
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT) # Loading the tokenizer
        
    tokenized_datasets = inshort_data.map(lambda df: tokenize_function(df,tokenizer), 
     batched=True, remove_columns=["text"]) #Tokenization of sentences


    ### Loading the model
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=7,
    label2id=label2id, id2label=id2label)
    
    ### Let's define the data collator for classification tasks
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    ### Tracking of experiment
    mlflow.set_experiment(experiment_name="NewsClassifer") # Name of the experience
    mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db") # Where to save the result

    with mlflow.start_run() as run :
        OUTPUT_MODEL_DIR= f'runs:/{run.info.run_id}/text-classifier'
    ### Preparation for trainings
        training_args = TrainingArguments(
            output_dir="./checkpoints",
            num_train_epochs=6,
            per_device_train_batch_size=96,
            per_device_eval_batch_size=96,
            weight_decay=1e-2,
            load_best_model_at_end=True,
            learning_rate=5e-6,
            do_predict=True,
            save_total_limit=2,
            save_strategy="epoch",
            eval_strategy="epoch",
            metric_for_best_model="loss",
            greater_is_better=False,
            do_eval=True
        ) # Training_args

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        ) # Trainer setup
        
        ### Training  model
        trainer.train()
        classification_pipeline = pipeline("text-classification", model=trainer.model, tokenizer=tokenizer)
        # Exemple d'entrée pour la signature
        input_example = ["This is a great movie!"]
        output_example = classification_pipeline(input_example)

        # Inférer la signature du modèle
        signature = infer_signature(input_example, output_example)
        ### Save model pipeline for inference
        model_info = mlflow.transformers.log_model(
            transformers_model=classification_pipeline,
            artifact_path="text-classifier",
            task="text-classification",
            signature=signature,
           input_example=input_example
    )
        print(f"✨ The run ID of the trained model is: {run.info.run_id} ✨")

