""" Implementing an API to test the model"""
from inference import predict
import uvicorn
from fastapi import FastAPI
import gradio as gr


app = FastAPI()

def make_prediction(prompt):
    """ Make the prediction """
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

#Mounting the gradio app
app = gr.mount_gradio_app(app, iface, path="/predict")

if __name__ == "__main__":
    uvicorn.run( app="api:app", host="0.0.0.0", port=8000 )
     #uvicorn.run(app="api:app", host=os.getenv("UVICORN_HOST"), port=int(os.getenv("UVICORN_PORT")))