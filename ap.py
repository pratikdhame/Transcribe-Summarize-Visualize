from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper
import torch
from pathlib import Path
import shutil
import os
import cv2
from transformers import pipeline
import yt_dlp

# Import from transcribe script
from transcribe import (
    generate_summary,
    extract_frames,
    detect_diagram_frames,
    transcribe as whisper_transcribe,
)

# Create FastAPI instance
app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
whisper_model = whisper.load_model("small", device=device)

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

# Persistent storage for files
PERSISTENT_DIR = Path("persistent_storage")
PERSISTENT_DIR.mkdir(exist_ok=True)

# Keywords for diagram detection
DIAGRAM_KEYWORDS = ["architecture", "diagram", "system design", "workflow", "process"]

class URLInput(BaseModel):
    url: str

@app.post("/transcribe/url")
async def transcribe_url(input: URLInput):
    try:
        # Create a subdirectory for the current session
        session_dir = PERSISTENT_DIR / "videos"
        session_dir.mkdir(parents=True, exist_ok=True)

        # Download video
        video_info = download(input.url, 360, session_dir)
        video_path = session_dir / video_info["filename"]

        # Transcribe
        transcript = whisper_transcribe(whisper_model, str(video_path), save=False)
        summary = generate_summary(transcript, summarizer)

        # Extract frames
        frames = extract_frames(str(video_path), num_frames=20)

        # Create frames output directory
        extracted_frames_dir = PERSISTENT_DIR / "extracted_frames"
        extracted_frames_dir.mkdir(parents=True, exist_ok=True)

        # Detect and save diagram frames
        video_id = os.path.splitext(video_info["filename"])[0]
        diagram_frames = detect_diagram_frames(
            frames,
            DIAGRAM_KEYWORDS,
            str(extracted_frames_dir),
            video_id,
        )

        return {
            "success": True,
            "transcription": transcript,
            "summary": summary,
            "title": video_info["title"],
            "diagram_frames": diagram_frames,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe/file")
async def transcribe_file(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_path = PERSISTENT_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Transcribe
        transcript = whisper_transcribe(whisper_model, str(file_path), save=False)
        summary = generate_summary(transcript, summarizer)

        # Extract frames
        frames = extract_frames(str(file_path), num_frames=20)

        # Create frames output directory
        extracted_frames_dir = PERSISTENT_DIR / "extracted_frames"
        extracted_frames_dir.mkdir(parents=True, exist_ok=True)

        # Detect and save diagram frames
        video_id = os.path.splitext(file.filename)[0]
        diagram_frames = detect_diagram_frames(
            frames,
            DIAGRAM_KEYWORDS,
            str(extracted_frames_dir),
            video_id,
        )

        return {
            "success": True,
            "transcription": transcript,
            "summary": summary,
            "title": file.filename,
            "diagram_frames": diagram_frames,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Download function for URL input
def download(url, resolution, videos_path):
    ydl_opts = {
        "format": f"bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]",
        "outtmpl": str(videos_path / "%(id)s.%(ext)s"),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        video_id = info_dict.get("id", None)
        title = info_dict.get("title", None)
        ext = info_dict.get("ext", "mp4")

        file_name = f"{video_id}.{ext}"
        file_path = videos_path / file_name

        return {
            "filename": file_name,
            "title": title,
            "filepath": file_path,
        }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
