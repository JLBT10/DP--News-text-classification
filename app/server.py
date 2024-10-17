# Implementing an API to test the news classification model
import uvicorn
from fastapi import FastAPI
import gradio as gr
import mlflow
from inference import predict
# Initialize FastAPI app
app = FastAPI()

# Set up the Gradio interface
iface = gr.Interface(
    fn=predict,
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