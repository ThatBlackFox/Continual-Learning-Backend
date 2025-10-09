from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
from fastapi.responses import StreamingResponse
from utils.models import *
from utils import train
import httpx
import threading
import asyncio

short_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8').strip()
train_messages = []
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

@app.post("/add-msg")
async def add_message(message: dict):
    train_messages.append(message['message'])
    return {"status": "Message added"}

@app.get("/api/train")
async def train_endpoint(batch_size,ewc_lambda,buffer_size,epochs,learning_rate,refresh_frequency,refresh_steps,task):
    train_thread = threading.Thread(target = train.train_loop, kwargs={"batch_size":int(batch_size),
                                                                       "ewc_lambda":float(ewc_lambda),
                                                                       "buffer_size":int(buffer_size),
                                                                       "epochs":int(epochs),
                                                                       "lr":float(learning_rate),
                                                                       "refresh_frequency":int(refresh_frequency),
                                                                       "refresh_steps":int(refresh_steps),
                                                                       "dataset":task})
    train_thread.start()
    async def event_stream():
        while True:
            if train_messages:
                message = train_messages.pop(0)
                if 'Complete' in message:
                    yield f"> {message}\n\n"
                    yield f"--------------------------------\n\n"
                    break
                yield f"> {message}\n\n"
            await asyncio.sleep(1)  # Adjust this to control the streaming rate

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/inference")
async def inference(sample: Smiles):
    ...
    #TODO add loading model and sending prediction logic
    return sample