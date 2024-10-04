
def convert_label_to_id(hf_df, label_to_id:dict):
    """ This functions has for goal to turn string labels into numbers
    
    Process a DatasetDict containing string labels.

    Parameters:
    - dataset (DatasetDict): A Hugging Face DatasetDict with the following properties:
        - Must contain at least one key (e.g., 'train', 'validation', etc.) that maps to a dataset.
        - Each dataset must have a column named 'labels'.
        - The 'labels' column must contain class names as strings (e.g., 'class_A', 'class_B', etc.).

    The function will perform the following operations:
    - Extract unique labels from the 'labels' column.
    - Create mappings from labels to integer IDs (label2id) and vice versa (id2label).
    - Convert string labels to numeric labels in the dataset for model training or processing.
    """
    
    hf_df["labels"] = label_to_id[hf_df["labels"]]  # Convert to ID
    
    return hf_df

    # Function to tokenize the input text
def tokenize_function(example,tokenizer):
        """Tokenize text"""
        return tokenizer(example["text"], truncation=True)
