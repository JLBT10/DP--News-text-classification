""" Training of the model """
# Import necessary libraries

#First party import
import dvc.api
import pandas as pd
import numpy as np  # Import before dataset

# Second party import
from transformers import (AutoModelForSequenceClassification,
AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer)
from sklearn.metrics import f1_score
from datasets import ClassLabel

#Third party import
from dataset import turns_pandas_into_HF_dataset, stratified_split_train_test

# Function to map labels to integers
def label2int(dataset):
    """ convert label into integer"""
    # Get the unique labels
    label_list = dataset.unique("labels")
    # Define a ClassLabel feature
    label_feature = ClassLabel(names=label_list)
    # Map the labels to integers
    dataset = dataset.map(lambda example: {"labels": label_feature.str2int(example["labels"])})
    # Assign the ClassLabel feature to the 'labels' column
    dataset = dataset.cast_column("labels", label_feature)
    return dataset, label_feature

# Function to create mapping of label to id and id to label
def label2id_id2label(label_feature):
    """ mapping label to id and id to label"""
    label2id_ = {v: k for k, v in enumerate(label_feature.names)}  # Mapping label to id
    id2label_ = {v: k for k, v in label2id.items()}  # Mapping id to label
    return label2id_, id2label_

# Function to compute metrics
def compute_metrics(p):
    """Overriding of the compute metrics so that it outs f1 score """
    # Extract predictions from the output
    preds = p.predictions if isinstance(p.predictions, tuple) else p.predictions
    # Convert predictions to class indices
    preds = np.argmax(preds, axis=1)
    # Compute F1 score
    f1 = f1_score(p.label_ids, preds, average='weighted')  # Use 'weighted' for multi-class
    return {"f1_score": f1}

if __name__ == '__main__':
    # Define the path to your dataset in the DVC-tracked repository
    DATASET_PATH = './data/inshort.csv'

    # Open the dataset file using dvc.api.open
    with dvc.api.open(DATASET_PATH , repo='https://github.com/JLBT10/NewsClassifier-BERT.git') as f:
        # Read the header first
        columns = f.readline().strip().split('|')
        inshort_data = pd.DataFrame(columns=columns)

        # Process each remaining line in the file
        for idx, line in enumerate(f):
            line_process = line.strip().split('|')
            if len(columns) == len(line_process):
                inshort_data.loc[idx] = line_process
            else:
                print(f"Line {idx} has mismatched columns: {line_process}")

    # Load data
    inshort_data = turns_pandas_into_HF_dataset(inshort_data)
    inshort_data = inshort_data.shuffle(10).select(range(100))

    # Process the label
    inshort_data, label = label2int(inshort_data)  # Turns str labels into id
    label2id, id2label = label2id_id2label(label)  # Mapping dictionaries

    # Split the dataset into train and test
    inshort_data = stratified_split_train_test(inshort_data)

    # Model name
    CHECKPOINT = "bert-base-cased"

    # Loading the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)


    # Function to tokenize the input text
    def tokenize_function(example):
        """Tokenize text"""
        return tokenizer(example["text"], truncation=True, padding=True)

    # Loading the model
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=7,
     label2id=label2id, id2label=id2label)

    # Tokenization of data
    tokenized_datasets = inshort_data.map(tokenize_function, batched=True)

    # Remove the "text" column
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])

    # Data Collator with padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        use_mps_device=True,
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
        evaluation_strategy="epoch",
        metric_for_best_model="loss",
        overwrite_output_dir=True,
        greater_is_better=False,
        do_eval=True,
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model("./runs/best_model")
