import mlflow
from transformers import pipeline

def predict(prompt):
    """
    Function to make predictions using the loaded model
    """
    # Create a text classification pipeline using the specified model
    RUN_ID = "d9b203e2fb4249a4991711db81898a1f"
    MODEL_PATH = f'mlruns/1/{RUN_ID}/artifacts/text-classifier'
    model = mlflow.pyfunc.load_model(MODEL_PATH)
    output = model.predict(prompt)
    
    # Return the prediction result
    return output.loc[0,"label"]
