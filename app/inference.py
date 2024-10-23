import mlflow
from transformers import pipeline

def predict(prompt):
    """
    Function to make predictions using the loaded model
    """
    # Create a text classification pipeline using the specified model
    RUN_ID = "a1b84b847fb445c8aa9fc021fb4a8581"
    MODEL_PATH = f'./mlruns/1/{RUN_ID}/artifacts/text-classifier'
    model = mlflow.pyfunc.load_model(MODEL_PATH)
    output = model.predict(prompt)
    
    # Return the prediction result
    return output.loc[0,"label"]
