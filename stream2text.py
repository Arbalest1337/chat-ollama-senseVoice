# uvicorn stream2text:app
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from audio2text import audio_to_text
import numpy as np
import soundfile as s
import os
import uuid

dir_audio = "audio"
os.makedirs(dir_audio, exist_ok=True)

app = FastAPI()


def generate_unique_filename(extension=".wav"):
    unique_id = str(uuid.uuid4())
    return unique_id + extension


@app.get("/index")
async def read_html_file():
    with open("stream.html", "r", encoding="utf-8") as file:
        content = file.read()
    return HTMLResponse(content)


@app.websocket("/audio_to_text")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_bytes()
            print(f"data {len(data)}")
            if len(data) % 2 != 0:
                data += b"\x00"
            audio_data = np.frombuffer(data, dtype=np.int16)
            file_name = f"{dir_audio}/{uuid.uuid4()}.webm"
            with open(file_name, "wb") as file:
                file.write(audio_data)
            text = audio_to_text(file_name)
            await websocket.send_text(f"{text}")
            os.remove(file_name)
        except Exception as e:
            print(f"WS Error: {e}")
            break
