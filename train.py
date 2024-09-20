# Import 
from transformers import (AutoModelForSequenceClassification,AutoTokenizer,
                        DataCollatorWithPadding, TrainingArguments, Trainer)
from sklearn.metrics import f1_score
from datasets import load_from_disk, ClassLabel

# Custom function for splitting train/test
from dataset import * 
import numpy as np

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding=True)

def label2int(dataset):

   # Get the unique labels
    label_list = dataset.unique("labels")

    # Define a ClassLabel feature
    label_feature = ClassLabel(names=label_list)

    # Map the labels to integers
    dataset = dataset.map(lambda example: {"labels": label_feature.str2int(example["labels"])})

    # Assign the ClassLabel feature to the 'labels' column
    dataset = dataset.cast_column("labels", label_feature)
    return dataset, label_feature

def label2id_id2label(label_feature):
    label2id = {v: k for k, v in enumerate(label_feature.names)} #Mapping label to id
    id2label = {v:k for k,v in label2id.items()} # Mapping id to label
    return label2id, id2label

def compute_metrics(p):
    # Extract predictions from the output
   
    preds = p.predictions if isinstance(p.predictions, tuple) else p.predictions
    
    # Convert predictions to class indices
    preds = np.argmax(preds, axis=1)
    print(p.label_ids,preds)
    # Compute F1 score
    f1 = f1_score(p.label_ids, preds, average='weighted')  # Use 'weighted' for multi-class, adjust as needed
   # print(p.label_ids,preds)
    return {"f1_score": f1}


if __name__ == '__main__':
        
    # Load data
    inshort_data = load_from_disk("./data/inshort_dataset")
    inshort_data = inshort_data.shuffle(10).select(range(100))
    #Process the label
    inshort_data, label = label2int(inshort_data) # Turns str labels into id
    label2id, id2label = label2id_id2label(label) # dictionnary mapping label to id and id to label

    # Split the dataset into train and test
    inshort_data = stratified_split_train_test(inshort_data)

    # Model name
    checkpoint = "bert-base-cased"  

    # Loading the tokenize
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Loading the model
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=7,label2id=label2id,id2label=id2label)

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
    eval_dataset=tokenized_datasets["test"],  # Uncomment if you have a validation set
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model("./runs/best_model")