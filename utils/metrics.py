from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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
    #compute_confusion_matrix(labels,predictions)
    # Afficher la matrice de confusion
    #plt.show()
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }