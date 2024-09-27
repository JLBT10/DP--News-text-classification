# Implementing an API to test the news classification model
from transformers import pipeline
import uvicorn
from fastapi import FastAPI
import gradio as gr
import mlflow
# Initialize FastAPI app
app = FastAPI()

def predict(prompt, path_to_model="/src/runs/best_model"):
    """
    Function to make predictions using the loaded model
    """
    # Create a text classification pipeline using the specified model
    
    run_id = "024e1d8189eb4c5abe4bc5107897b3f3"
    model_uri = f"run:/{run_id}/model"
    model = mlflow.pytorch. load_model(model_uri)
    pipe = pipeline("text-classification", model=path_to_model)
    # Make a prediction
    output = pipe(prompt)
    
    # Convert the score to a percentage and round to 2 decimal places
    output[0]["score"] = f"{round(output[0]['score'] * 100, 2)}%"
    
    # Return the prediction result
    return {"answer": output[0]}

def make_prediction(prompt):
    """ Wrapper function to make the prediction """
    return predict(prompt)

# Set up the Gradio interface
iface = gr.Interface(
    fn=make_prediction,
    inputs=gr.Textbox(label="Enter News Prompt"),
    outputs=gr.Textbox(label="Predicted Category"),
    title="News Category Classifier",
    description="Enter a news article prompt to get its predicted category."
)

@app.get('/')
def read_root():
    """ Welcome page of the API """
    return {'message': 'Welcome to the model API, to access the interface go to localhost:8000/predict .'}

# Mount the Gradio app to the FastAPI app
app = gr.mount_gradio_app(app, iface, path="/predict")

if __name__ == "__main__":
    # Run the FastAPI app using Uvicorn
    uvicorn.run(app="api:app", host="0.0.0.0", port=8000)
    
    # Alternative way to run the app using environment variables
    # uvicorn.run(app="api:app", host=os.getenv("UVICORN_HOST"), port=int(os.getenv("UVICORN_PORT")))