import pandas as pd
from datasets import Dataset
import os
import glob
import pandas as pd
import glob
import os

def create_inshort_dataset(path):
    # Initialisation of variables
    inshort_dataset = pd.DataFrame()

    # Get the list of the csv data inside the directory
    csv_list = glob.glob(os.path.join(path, "*.csv"))
    for file in csv_list:
        data = pd.read_csv(file)
        inshort_dataset = pd.concat([inshort_dataset, data], axis=0)

    # Creating the input for our model
    inshort_dataset["text"] = inshort_dataset["news_headline"] + " " + inshort_dataset["news_article"]

    # Define the columns to clean
    columns_to_clean = ["news_headline", "news_article", "text", "labels"]

    # Clean the specified columns
    for column in columns_to_clean:
        if column in inshort_dataset.columns:
            inshort_dataset[column] = (
                inshort_dataset[column]
                .str.strip()  # Remove leading and trailing spaces
                .str.replace(r'\s+', ' ', regex=True)  # Replace multiple spaces with a single space
            )

    # Rename the target column
    inshort_dataset = inshort_dataset.rename(columns={'news_category': 'labels'})

    # Drop the 'Unnamed: 0' column if it exists
    if "Unnamed: 0" in inshort_dataset.columns:
        inshort_dataset = inshort_dataset.drop(["Unnamed: 0"], axis=1)

    # Replace newline characters with spaces
    #inshort_dataset = inshort_dataset.replace(r'\n', ' ', regex=True)

    # Save the cleaned dataset to a CSV file
    inshort_dataset.to_csv("./data/inshort.csv", index=False, sep='|')


    ####
    #Turns the pandas dataframe into a dataset format
    #inshort_dataset = turns_pandas_into_HF_dataset(inshort_dataset)
    #Save it in the data direcotry
    #inshort_dataset.save_to_disk('./data/inshort_dataset')
  
    return inshort_dataset

def turns_pandas_into_HF_dataset(pandas_df):
    # Convert Pandas DataFrame to Hugging Face Dataset
    hf_dataset = Dataset.from_pandas(pandas_df)

    # Optional: Remove the index column if it was included
    hf_dataset = hf_dataset.remove_columns(["__index_level_0__",'news_headline','news_article'])
    return hf_dataset
    

def stratified_split_train_test(dataset):
    
    dataset = dataset.train_test_split(test_size=0.3,shuffle=True, stratify_by_column="labels")
    return dataset


"""def load_with_dvc(dataset_path, repo):
    # Open the dataset file using dvc.api.open
    with dvc.api.open(dataset_path , repo=repo) as f:
        # Read the header first   
        columns = f.readline().strip().split('|')
        dataset = pd.DataFrame(columns=columns)

        # Process each remaining line in the file
        for idx, line in enumerate(f):
            line_process = line.strip().split('|')
            if len(columns) == len(line_process):
                dataset.loc[idx] = line_process
            else:
                print(f"Line {idx} has mismatched columns: {line_process}")

    return dataset
#if __name__ == "__main__" :
    #inshort_data = create_inshort_dataset("./data")
    #print(inshort_data)"""