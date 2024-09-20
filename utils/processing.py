import pandas as pd
from datasets import Dataset
import os
import glob

def load_inshort_data(path):
    # Initiatisation of variables
    inshort_dataset=pd.DataFrame()

    # Get the list of the csv data inside the repertory
    csv_list = glob.glob(os.path.join(path,"*.csv"))
    for file in csv_list:
        data = pd.read_csv(file)
        inshort_dataset = pd.concat([inshort_dataset,data],axis=0)

    inshort_dataset["text"] = inshort_dataset["news_headline"] + " " + inshort_dataset["news_headline"]
    inshort_dataset.rename(columns={'news_category': 'labels'})
    inshort_dataset = turns_pandas_into_HF_dataset(inshort_dataset)
  
    return inshort_dataset

def turns_pandas_into_HF_dataset(pandas_df):
    # Convert Pandas DataFrame to Hugging Face Dataset
    hf_dataset = Dataset.from_pandas(pandas_df)

    # Optional: Remove the index column if it was included
    hf_dataset = hf_dataset.remove_columns(["__index_level_0__",'Unnamed: 0','news_headline','news_headline'])
    return hf_dataset

if __name__ == "__main__" :

    inshort_data = load_inshort_data("./data")
    inshort_data.save_to_disk('./data/inshort_dataset')