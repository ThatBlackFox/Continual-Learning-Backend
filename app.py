from fastapi import FastAPI
import subprocess
from fastapi.responses import StreamingResponse
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

@app.get("/train")
async def train():
    return StreamingResponse(train(), media_type="text/plain")

@app.post("/inference")
async def inference(sample: Smiles):
    ...
    #TODO add loading model and sending prediction logic
    return sample