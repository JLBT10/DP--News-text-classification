import uvicorn
from fastapi import FastAPI
import gradio as gr
import argparse
from inference import predict  # Import the predict function

# Initialize FastAPI app
app = FastAPI()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, required=True, help='Model ID for prediction')
    return parser.parse_args()

# Parse command line arguments
opt = parse_opt()

# Set up the Gradio interface
def main():
    iface = gr.Interface(
        fn=lambda prompt: predict(prompt, run_id=opt.run_id),  # Pass model_id to predict function
        inputs=gr.Textbox(label="Enter News Prompt"),
        outputs=gr.Textbox(label="Predicted Category"),
        title="News Category Classifier",
        description="Enter a news article prompt to get its predicted category."
    )
    return iface

@app.get('/')
def read_root():
    """ Welcome page of the API """
    return {'message': 'Welcome to the model API, to access the interface go to Public-IP-EC2:8000/predict.'}

# Mount the Gradio app to the FastAPI app
app = gr.mount_gradio_app(app, main(), path="/predict")

if __name__ == "__main__":
    # Run the FastAPI app using Uvicorn
    uvicorn.run(app="server:app", host="0.0.0.0", port=8000)
