import mlflow
from transformers import pipeline

def predict(prompt):
    """
    Function to make predictions using the loaded model
    """
    # Create a text classification pipeline using the specified model
    
    RUN_ID = "b11f5c9401304a718ba2a6462ef5a7d7"
    model_path = f'/mlflow/artifacts/1/{RUN_ID}/artifacts/text-classifier'
    model = mlflow.pyfunc.load_model(model_path)
    output = model.predict(prompt)
    
    # Return the prediction result
    return output.loc[0,"label"]
