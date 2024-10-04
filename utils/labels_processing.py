def map_label_to_id(label_list: list):
    """
    Map labels to unique integer IDs.

    Parameters:
    - label_list (list): A list of string labels (e.g., ['class_A', 'class_B']).

    Returns:
    - dict: A dictionary mapping each label to its corresponding ID.
    
    Example:
    >>> label_to_id(['cat', 'dog', 'fish'])
    {'cat': 0, 'dog': 1, 'fish': 2}
    """
    return {v: k for k, v in enumerate(label_list)}


def map_id_to_label(label_to_id_dict:dict):

    """
    Map integer IDs back to their corresponding labels.

    Parameters:
    - label_to_id_dict (dict): A dictionary mapping labels to IDs (e.g., {'class_A': 0, 'class_B': 1}).

    Returns:
    - dict: A dictionary mapping each ID back to its corresponding label.
    
    Example:
    id_to_label({'cat': 0, 'dog': 1, 'fish': 2})
    {0: 'cat', 1: 'dog', 2: 'fish'}
    """

    return {v: k for k, v in label_to_id_dict.items()} 

def get_label2id_id2label(label_list:list):
    label_to_id = map_label_to_id(label_list)
    id_to_label = map_id_to_label(label_to_id)
    return label_to_id, id_to_label 
