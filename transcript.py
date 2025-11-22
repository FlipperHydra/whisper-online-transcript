from typing import Optional
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import whisper
import os

import shutil

model = whisper.load_model("turbo")

def stt(filename: str):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Audio file not found: {filename}")
    
    result = model.transcribe(filename)
    return result["text"]


app = FastAPI()

#CONNECTION TO FRONT 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return FileResponse("index.html")

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    transcription = stt(file_path)
    os.remove(file_path)
    
    return {"text": transcription, "filename": file.filename}
#CONNECTION TO FRONT

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        #reload=True
        )

