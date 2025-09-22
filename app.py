from fastapi import FastAPI, WebSocket
import subprocess
from fastapi.responses import HTMLResponse
from utils.models import *

short_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8').strip()

model_info = {
    "Model": "ChemBERTa-zinc-base-v1",
    "Training modes": ["TIL"],
    "Datasets": ["BBBP","Bitter", "Sweet"]
}

app = FastAPI()

@app.get("/")
async def root():
    return {"message": f"Welcome to the model training server | on commit - {short_hash} | https://github.com/ThatBlackFox/Continual-Learning-Backend"}

@app.get("/connect")
async def connect():
    return model_info

@app.websocket("/train")
async def train(websocket: WebSocket):
    await websocket.accept()
    still_traing = True
    await websocket.send_text(f"Connected to training server | on commit - {short_hash}")
    while still_traing:
        ...
        #TODO: Add training logic and a way to send back logs

@app.post("/inference")
async def inference(sample: Smiles):
    ...
    #TODO add loading model and sending prediction logic
    return sample