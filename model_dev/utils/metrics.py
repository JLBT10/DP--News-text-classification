from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import numpy as np

# Function to compute metrics
def compute_metrics(eval_preds):
    # Unpack the logits and labels from eval_preds
    logits, labels = eval_preds  # logits shape: (num_instances, num_classes)
    
    # Get predictions by taking the argmax over the logits
    predictions = np.argmax(logits, axis=-1)  # shape: (num_instances,)
    
    # Calculate accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    conf_matrix = confusion_matrix(labels,predictions)
    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(labels), yticklabels=np.unique(labels))
    plt.title("Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save confusion matrix as an image
    os.makedirs("./model_dev/artifacts", exist_ok=True)
    confusion_matrix_path = os.path.join("./model_dev/artifacts", "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    plt.close()
    # Afficher la matrice de confusion
    #plt.show()
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os

def compute_metrics_n(eval_preds, output_dir="./results"):
    # Unpack the logits and labels from eval_preds
    logits, labels = eval_preds  # logits shape: (num_instances, num_classes)
    
    # Get predictions by taking the argmax over the logits
    predictions = np.argmax(logits, axis=-1)  # shape: (num_instances,)
    
    # Calculate accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

    # Generate confusion matrix
    conf_matrix = confusion_matrix(labels, predictions)

    # Plot confusion matrix using plt.imshow
    plt.figure(figsize=(10, 7))
    plt.imshow(conf_matrix, interpolation='nearest', cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()  # Add a colorbar to the side
    
    # Set ticks and labels for x and y axes
    classes = np.unique(labels)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Annotate the confusion matrix with counts
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, str(conf_matrix[i, j]), horizontalalignment="center", color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Save confusion matrix as an image
    os.makedirs(output_dir, exist_ok=True)
    confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    plt.close()

    # Return metrics
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix_path': confusion_matrix_path  # Path to saved confusion matrix
    }