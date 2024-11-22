from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from googletrans import Translator
import whisper
import torch
from pathlib import Path
import shutil
import os
import cv2
from transformers import pipeline
import yt_dlp
from pytesseract import image_to_string, pytesseract, Output
import numpy as np

# Specify the Tesseract executable path
pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Create FastAPI instance
app = FastAPI()
translator = Translator()
# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles

# Mount the persistent storage directory for serving static files
app.mount("/persistent_storage", StaticFiles(directory="persistent_storage"), name="persistent_storage")

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
whisper_model = whisper.load_model("small", device=device)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

# Persistent storage for files
PERSISTENT_DIR = Path("persistent_storage")
PERSISTENT_DIR.mkdir(exist_ok=True)

# Keywords for diagram detection
DIAGRAM_KEYWORDS = ["architecture", "diagram", "system design", "workflow", "process"]

class URLInput(BaseModel):
    url: str

class TranslationRequest(BaseModel):
    text: str
    language: str

@app.post("/translate")
async def translate_text(request: TranslationRequest):
    try:
        translated = translator.translate(request.text, dest=request.language)
        return {"success": True, "translatedText": translated.text}
    except Exception as e:
        return {"success": False, "error": str(e)}
    
# Frame preprocessing
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

# Detect diagram frames
def detect_diagram_frames(frames, keywords):
    diagram_frames = []
    for idx, frame in enumerate(frames):
        try:
            processed_frame = preprocess_frame(frame)
            text_data = image_to_string(processed_frame, output_type=Output.DICT)
            detected_text = text_data["text"].lower()
            contains_keywords = any(keyword in detected_text for keyword in keywords)
            edges = cv2.Canny(processed_frame, 50, 150)
            edge_density = edges.sum()
            if contains_keywords or edge_density > 2000000:
                diagram_frames.append(frame)
        except Exception as e:
            print(f"Error processing frame {idx}: {e}")
    return diagram_frames

# Extract frames from video
def extract_frames(video_path, num_frames=10):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for frame_index in frame_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = video.read()
        if ret:
            frames.append(frame)
    video.release()
    return frames

# Transcribe and summarize video
def whisper_transcribe(model, video_path, save=False):
    result = model.transcribe(video_path)
    return result["text"]

def generate_summary(transcript, summarizer):
    max_chunk = 1024
    chunks = [transcript[i:i + max_chunk] for i in range(0, len(transcript), max_chunk)]
    summaries = [summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]["summary_text"] for chunk in chunks]
    return " ".join(summaries)

@app.post("/transcribe/url")
async def transcribe_url(input: URLInput):
    try:
        session_dir = PERSISTENT_DIR / "videos"
        session_dir.mkdir(parents=True, exist_ok=True)

        video_info = download(input.url, 360, session_dir)
        video_path = session_dir / video_info["filename"]

        transcript = whisper_transcribe(whisper_model, str(video_path), save=False)
        summary = generate_summary(transcript, summarizer)
        frames = extract_frames(str(video_path), num_frames=20)

        extracted_frames_dir = PERSISTENT_DIR / "extracted_frames"
        extracted_frames_dir.mkdir(parents=True, exist_ok=True)

        diagram_frames = detect_diagram_frames(frames, DIAGRAM_KEYWORDS)
        video_id = os.path.splitext(video_info["filename"])[0]

        for idx, frame in enumerate(diagram_frames):
            frame_path = extracted_frames_dir / f"{video_id}_diagram_{idx}.jpg"
            cv2.imwrite(str(frame_path), frame)

        return {
            "success": True,
            "transcription": transcript,
            "summary": summary,
            "title": video_info["title"],
            "diagram_frames": [str(frame) for frame in extracted_frames_dir.glob(f"{video_id}_diagram_*.jpg")],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe/file")
async def transcribe_file(file: UploadFile = File(...)):
    try:
        file_path = PERSISTENT_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        transcript = whisper_transcribe(whisper_model, str(file_path), save=False)
        summary = generate_summary(transcript, summarizer)
        frames = extract_frames(str(file_path), num_frames=20)

        extracted_frames_dir = PERSISTENT_DIR / "extracted_frames"
        extracted_frames_dir.mkdir(parents=True, exist_ok=True)

        diagram_frames = detect_diagram_frames(frames, DIAGRAM_KEYWORDS)
        video_id = os.path.splitext(file.filename)[0]

        for idx, frame in enumerate(diagram_frames):
            frame_path = extracted_frames_dir / f"{video_id}_diagram_{idx}.jpg"
            cv2.imwrite(str(frame_path), frame)

        return {
            "success": True,
            "transcription": transcript,
            "summary": summary,
            "title": file.filename,
            "diagram_frames": [str(frame) for frame in extracted_frames_dir.glob(f"{video_id}_diagram_*.jpg")],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        return {"filename": file_name, "title": title, "filepath": videos_path / file_name}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
