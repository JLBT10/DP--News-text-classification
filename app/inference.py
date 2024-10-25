import mlflow
from transformers import pipeline
import argparse

model_cache = {}

def load_model(run_id: str) -> any:
    if run_id not in model_cache:
        try:
            MODEL_PATH = f'/Users/jlbt/boa_workspace/NewsClassifier-BERT/model_dev/mlruns/1/{run_id}/artifacts/text-classifier'
            model_cache[run_id] = mlflow.pyfunc.load_model(MODEL_PATH)
        except Exception as e:
            # Handle model loading exceptions
            return str(e)
    return model_cache[run_id]

def predict(prompt: str, run_id: str) -> str:
    model = load_model(run_id)
    output = model.predict(prompt)
    return output.loc[0, "label"]

