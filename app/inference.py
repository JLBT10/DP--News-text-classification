import mlflow
from transformers import pipeline

def predict(prompt, path_to_model="/src/runs/best_model"):
    """
    Function to make predictions using the loaded model
    """
    # Create a text classification pipeline using the specified model
    
    run_id = "024e1d8189eb4c5abe4bc5107897b3f3"
    model_path = f'mlruns/artifacts/{run_id}/text-classifier'
    model = mlflow.pyfunc.load_model(model_path )
    output = model.predict(prompt)
    pipe = pipeline("text-classification", model=path_to_model)
    # Make a prediction
    output = pipe(prompt)
    
    # Return the prediction result
    return output.loc[0,"labels"]