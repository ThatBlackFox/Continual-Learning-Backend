from fastapi import FastAPI
import subprocess
from fastapi.responses import StreamingResponse
from utils.models import *
from utils import til_train, dil_train, cil_train
import threading
import asyncio
from utils import infer

short_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8').strip()
train_messages = []
model_info = {
    "Model": "ChemBERTa-zinc-base-v1",
    "Training modes": ["TIL"],
    "Datasets": ["BBBP", "Bitter", "Sweet"]
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
async def train_endpoint(batch_size,ewc_lambda,buffer_size,epochs,learning_rate,refresh_frequency,refresh_steps,task,train_mode):
    kwargs = {  "batch_size":int(batch_size),
                "ewc_lambda":float(ewc_lambda),
                "buffer_size":int(buffer_size),
                "epochs":int(epochs),
                "lr":float(learning_rate),
                "refresh_frequency":int(refresh_frequency),
                "refresh_steps":int(refresh_steps),
                "dataset":task}
    if train_mode == 'til':
        train_thread = threading.Thread(target = til_train.train_loop, kwargs=kwargs)
    elif train_mode == 'dil':
        train_thread = threading.Thread(target = dil_train.train_loop, kwargs=kwargs)
    else:
        train_thread = threading.Thread(target = cil_train.train_loop, kwargs=kwargs)
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

@app.post("/api/inference")
async def inference(sample: Smiles):
    print(sample.text, sample.dataset)
    output = infer.inference(sample.text, sample.dataset)
    return output