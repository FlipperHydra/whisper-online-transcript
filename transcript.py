from typing import Optional
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import whisper
import os
model = whisper.load_model("turbo")

#Transcribes
def stt(filename: str):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Audio file not found: {filename}")
    
    result = model.transcribe(filename)
    return result["text"]

#Create instance of the FastAPI class
app = FastAPI()

#Allows browser to sync backend and frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#CONNECT TO FRONTEND
@app.get("/")
def read_root():
    return FileResponse("index.html")

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    #Splits into 1Mb chunks
    file_path = f"temp_{file.filename}" 
    with open(file_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):  
            buffer.write(chunk)
    
    transcription = stt(file_path)
    os.remove(file_path)
    
    return {"text": transcription, "filename": file.filename}
#CONNECT TO FRONTEND

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        timeout_keep_alive=4800  
    )

#If suggestions for efficiency, email: doe48944067@gmail.com