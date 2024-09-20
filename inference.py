
""" This files is to implement inferences  with the trained model"""
from transformers import pipeline

def predict(prompt,path_to_model="./../runs/best_model"):  

    pipe = pipeline("text-classification", model=path_to_model)
    output = pipe(prompt)
    output[0]["score"] = f"{round(output[0]['score'] * 100, 2)}%"
    
    return {"answer" : output[0]}