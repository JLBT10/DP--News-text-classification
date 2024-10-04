from datasets import DatasetDict, Dataset

def select_n_rows(hf_dataset: DatasetDict | Dataset ,nrow_to_keep:int)-> DatasetDict | Dataset :
    """
    This function allows you to select a number of rows from a DatasetDict or Dataset.

    Example: 
    print(datasets)

    DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    }),
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    }),
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
    })
    
    Parameters:
    hf_dataset (DatasetDict or Dataset): a DatasetDict or Dataset
    nrow_to_keep (int): specifies the number of rows to keep
    
    Returns:
    A dataset in Hugging Face format with the specified number of rows.
    
    Exceptions:
    TypeError: If hf_dataset is neither a DatasetDict nor a Dataset.

    """

    if isinstance(hf_dataset, DatasetDict):
        # Create a new DatasetDict to store the reduced datasets
        reduced_datasets = DatasetDict({
            split: hf_dataset[split].select(range(min(nrow_to_keep, len(hf_dataset[split]))))
            for split in hf_dataset.keys()
        })
        return reduced_datasets
    elif isinstance(hf_dataset, Dataset):
        # Si c'est un dataset unique, sélectionner les lignes directement
        return hf_dataset.select(range(min(nrow_to_keep, len(hf_dataset))))
    else:
        # Lever une exception si l'entrée n'est ni DatasetDict ni Dataset
        raise TypeError("hf_dataset doit être un DatasetDict ou un Dataset du package Hugging Face.")


