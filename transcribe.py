import os
import json
import re
import argparse
import cv2
import numpy as np
import torch
import whisper
import pytube
import yt_dlp
from pytesseract import image_to_string, Output
from transformers import pipeline

def extract_video_id(url):
    pattern = r'(?:https?:\/\/)?(?:www\.)?youtu(?:\.be|be\.com)\/(?:.*v(?:\/|=)|(?:.*\/)?)([\w\-]+)'
    match = re.match(pattern, url)
    if match:
        return match.group(1)
    return None

def load_urls(filename):
    with open(filename, 'r') as f:
        urls = json.load(f)
    return urls

def download(url, resolution, videos_path):
    ydl_opts = {
        'format': f'bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]',
        'outtmpl': os.path.join(videos_path, '%(id)s.%(ext)s'),
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        video_id = info_dict.get('id', None)
        title = info_dict.get('title', None)
        ext = info_dict.get('ext', 'mp4')
        
        file_name = f"{video_id}.{ext}"
        file_path = os.path.join(videos_path, file_name)
        
        return {
            "filename": file_name,
            "title": title,
            "filepath": file_path
        }

def preprocess_frame(frame):
    """Preprocess frame for OCR and diagram detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def detect_diagram_frames(frames, keywords, output_dir, video_id):
    """Detect and save diagram-related frames."""
    diagram_frames = []
    
    for idx, frame in enumerate(frames):
        # Preprocess the frame for better OCR
        processed_frame = preprocess_frame(frame)

        # Extract text using OCR
        try:
            text_data = image_to_string(processed_frame, output_type=Output.DICT)
            detected_text = text_data.get("text", "").lower()
        except Exception as e:
            print(f"OCR Error: {e}")
            detected_text = ""

        # Check for keywords in detected text
        contains_keywords = any(keyword in detected_text for keyword in keywords)

        # Edge detection for diagrams
        edges = cv2.Canny(processed_frame, 50, 150)
        edge_density = edges.sum()

        # Criteria for diagram frames
        if contains_keywords or edge_density > 2000000:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save frame
            frame_filename = f"{video_id}_diagram_{idx}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            diagram_frames.append(frame_path)

    return diagram_frames

def extract_frames(video_path, num_frames=20):
    """Extract frames from video."""
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return []

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        print(f"Error: Video file {video_path} has no frames.")
        video.release()
        return []
    
    # Use numpy to select evenly distributed frames
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for frame_index in frame_indices:
        if frame_index < 0 or frame_index >= total_frames:
            print(f"Skipping invalid frame index: {frame_index}")
            continue

        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = video.read()
        if ret:
            frames.append(frame)
        else:
            print(f"Failed to read frame at index {frame_index}")
    
    video.release()
    return frames

def transcribe(model, video_path, save):
    print("Transcribing", video_path)
    result = model.transcribe(video_path)
    text = [item["text"] for item in result["segments"]]
    text = "".join(text)
    if save:
        os.remove(video_path)
    return text

def generate_summary(text, summarizer):
    max_chunk_length = 1024
    chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    
    return " ".join(summaries)

if __name__ == "__main__":
    # Paths
    input_path = "data/urls.json"
    videos_path = "data/videos"
    output_path = "data/output.json"
    frames_path = "data/extracted_frames"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    model_name = "small"
    whisper_model = whisper.load_model(model_name, device=device)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

    # Keywords for diagram detection
    keywords = ["architecture", "diagram", "system design", "workflow", "process"]

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=False, help="Single url method", default=None)
    parser.add_argument("--playlist", type=str, required=False, help="Playlist url method", default=None)
    parser.add_argument("--res", type=int, required=False, help="The resolution of the video(s) to download", default=360)
    parser.add_argument('--no-save', action='store_false', dest="save",
                        help='Add this to remove the video(s) from the local storage after transcription.')
    parser.add_argument('--local', action='store_true', dest="local",
                        help='Add this to use local files instead of downloading from youtube.')

    args = parser.parse_args()

    # Determine URLs
    if args.url:
        print("Option: from single url")
        urls = [args.url]
    elif args.playlist:
        print("Option: from playlist")
        urls = pytube.Playlist(args.playlist).video_urls
    elif args.local:
        print("Option: from local files")
        urls = os.listdir(videos_path)
    else:
        print("Option: from urls.json")
        urls = load_urls(input_path)

    # Process videos
    data = {}
    for file_name in urls:
        # Determine video source
        if args.local:
            video = {"filename": file_name, "title": file_name, "filepath": os.path.join(videos_path, file_name)}
        else:
            video = download(file_name, args.res, videos_path)

        # Transcribe video
        transcript = transcribe(whisper_model, video["filepath"], args.save)
        summary = generate_summary(transcript, summarizer)

        # Extract frames
        try:
            frames = extract_frames(video["filepath"], num_frames=20)
            video_id = os.path.splitext(video["filename"])[0]
            diagram_frames = detect_diagram_frames(
                frames, 
                keywords, 
                frames_path, 
                video_id
            )
        except Exception as e:
            print(f"Frame extraction error for {video['filename']}: {e}")
            diagram_frames = []

        # Store data
        data[file_name] = {
            "title": video["title"],
            "transcription": transcript,
            "summary": summary,
            "diagram_frames": diagram_frames
        }

    # Save output to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)