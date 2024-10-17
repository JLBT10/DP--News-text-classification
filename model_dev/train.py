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
from datasets import ClassLabel, load_from_disk

# Project-Specific imports
from utils.metrics import compute_metrics
from utils.general import select_n_rows
from utils.labels_processing import *
from utils.map_functions import *

if __name__ == '__main__':
    ### Defining constant
    DATASET_PATH = './model_dev/data/inshort_dataset' #Dataset path on github
    CHECKPOINT = "bert-base-cased" # Name of the model
    OUTPUT_MODEL_DIR = "./model_dev/runs/best_model"

    ### Loading data
    inshort_data = load_from_disk(DATASET_PATH)

    
    ### Reducing the number of rows to process data faster for testing
    inshort_data = select_n_rows(inshort_data,100)

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
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=4,
    label2id=label2id, id2label=id2label)
    
    ### Let's define the data collator for classification tasks
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    ### Tracking of experiment
    mlflow.set_experiment(experiment_name="NewsClassifer") # Name of the experience
    mlflow.set_tracking_uri("sqlite:///model_dev/mlruns/mlflow.db") # Where to save the result

    with mlflow.start_run() as run :
    ### Preparation for trainings
        training_args = TrainingArguments(
            output_dir="./checkpoints",
            use_mps_device=False,
            num_train_epochs=1,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            weight_decay=1e-2,
            logging_dir="./save_model/logs",
            load_best_model_at_end=True,
            learning_rate=5e-6,
            do_predict=True,
            save_total_limit=2,
            save_strategy="epoch",
            eval_strategy="epoch",
            metric_for_best_model="loss",
            overwrite_output_dir=True,
            greater_is_better=False,
            do_eval=True,
            logging_steps=2
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

        os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
        trainer.save_model(OUTPUT_MODEL_DIR)
        mlflow.log_artifact(OUTPUT_MODEL_DIR,"M")
        ### Get model signature ready
        classification_pipeline = pipeline(model=OUTPUT_MODEL_DIR, task='text-classification')
        input_example = ["Facebook is a huge platform"]
        output = classification_pipeline(input_example)

        ### Model signature
        signature = mlflow.models.infer_signature(input_example, output)

        ### Save model pipeline for inference
        model_info = mlflow.transformers.log_model(
            transformers_model=classification_pipeline,
            artifact_path="text-classifier",
            task="text-classification",
            signature=signature,
            input_example=input_example,
        )
