from datasets import load_from_disk
from evaluate import evaluator
from transformers import pipeline
import evaluate
from sklearn.metrics import confusion_matrix
import numpy as np

# Load the dataset from disk and shuffle it
data = load_from_disk("./data/inshort_dataset").shuffle(seed=42).select(range(100))

# Define the label mapping from strings to integers
label_mapping = {
    "technology": 0,
    "world": 1,
    "entertainment": 2,
    "politics": 3,
    "automobile": 4,
    "science": 5,
    "sports": 6
}

# Function to convert string labels into integers
def convert_labels(example):
    example['label'] = label_mapping.get(example['labels'], -1)  # Convert the label, use -1 if the label is unknown
    return example

# Apply the function to the dataset to convert labels
data = data.map(convert_labels)

# Create the text-classification pipeline
pipe = pipeline(
    "text-classification",  # Define task as text classification
    model="./runs/best_model",  # Use the model saved in the specified directory
)

# Create the evaluator for text classification tasks
task_evaluator = evaluator("text-classification")

# Run evaluation with accuracy metric
eval_results = task_evaluator.compute(
    input_column="text",  # Column in the dataset with input text
    label_column="label",  # Column in the dataset with labels (now integers)
    model_or_pipeline=pipe,  # The pipeline with the model to be evaluated
    data=data,  # The dataset for evaluation
    metric="accuracy",  # The metric to evaluate (accuracy)
    label_mapping=label_mapping  # Map predicted labels to actual labels
)

print('###### ACCURACY ######')
# Print the results of the accuracy evaluation
print(eval_results, '\n')

# Set metric argument for future evaluations (weighted average for multiclass metrics)
task_evaluator.METRIC_KWARGS = {"average": "weighted"}

# Run evaluation with recall, precision, and F1 score metrics
eval_results = task_evaluator.compute(
    input_column="text",  
    label_column="label",  
    model_or_pipeline=pipe, 
    data=data,  
    metric=evaluate.combine(["recall", "precision", "f1"]),  # Combine recall, precision, and F1 score metrics
    label_mapping=label_mapping 
)


print('###### RECALL / PRECISION / F1 ######')
# Print the results of recall, precision, and F1 score evaluation
print(eval_results,'\n')



print('###### CONFUSION MATRIX ######')
# Get true labels and predictions
true_labels = data['label']
predictions = []

# Generate predictions using the model pipeline
for text in data['text']:
    pred = pipe(text)[0]['label']
    pred_int = int(label_mapping[pred])  # Convert string prediction back to integer
    predictions.append(pred_int)

# 5. Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels, predictions)
print("Confusion Matrix:")
print(conf_matrix)