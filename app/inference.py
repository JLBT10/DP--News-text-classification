import mlflow
from transformers import pipeline

def predict(prompt):
    """
    Function to make predictions using the loaded model
    """
    # Create a text classification pipeline using the specified model
    
    RUN_ID = "53ab23287a0b4a7090b645a93e44c4f7"
    model_path = f'/mlflow/artifacts/2/{RUN_ID}/artifacts/text-classifier'
    model = mlflow.pyfunc.load_model(model_path)
    output = model.predict(prompt)
    
    # Return the prediction result
    return output.loc[0,"label"]