from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
# Function to compute train metrics
def compute_metrics(eval_preds):
    # Unpack the logits and labels from eval_preds
    logits, labels = eval_preds  # logits shape: (num_instances, num_classes)
    
    # Get predictions by taking the argmax over the logits
    predictions = np.argmax(logits, axis=-1)  # shape: (num_instances,)
    
    # Calculate accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


    # Function to compute eval metrics
def eval_compute_metrics(eval_preds):
    # Unpack the logits and labels from eval_preds
    logits, labels = eval_preds  # logits shape: (num_instances, num_classes)

    #For confusion matrix tick labels
    id2label= {
    0: "technology",
    1: "sports",
    2: "world",
    3: "politics",
    4: "entertainment",
    5: "automobile",
    6: "science"
  }
    cm_labels= [id2label[i] for i in np.unique(labels)]

    # Get predictions by taking the argmax over the logits
    predictions = np.argmax(logits, axis=-1)  # shape: (num_instances,)
    
    # Calculate accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    conf_matrix = confusion_matrix(labels,predictions)
    # Plot confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=cm_labels, yticklabels=cm_labels)
    plt.title("Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save confusion matrix as an image
    os.makedirs("./mlruns/artifacts", exist_ok=True)
    confusion_matrix_path = os.path.join("./mlruns/artifacts", "confusion_matrix.png")
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

