import cv2
import numpy as np
from pytesseract import image_to_string, pytesseract, Output

# Specify the Tesseract executable path (update it to match your installation path)
pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Function to preprocess frames for better OCR
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Increase contrast and binarize the image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

# Detect frames with relevant keywords or diagram-like features
def detect_diagram_frames(frames, keywords):
    diagram_frames = []
    for idx, frame in enumerate(frames):
        try:
            # Preprocess the frame for better OCR
            processed_frame = preprocess_frame(frame)

            # Extract text using OCR
            text_data = image_to_string(processed_frame, output_type=Output.DICT)
            detected_text = text_data["text"].lower()

            # Check for keywords in detected text
            contains_keywords = any(keyword in detected_text for keyword in keywords)

            # Edge detection for diagrams
            edges = cv2.Canny(processed_frame, 50, 150)
            edge_density = edges.sum()

            # Debug information
            print(f"Frame {idx}:")
            print(f"- Detected text: {detected_text}")
            print(f"- Contains keywords: {contains_keywords}")
            print(f"- Edge density: {edge_density}")

            # Add frame if it contains keywords or has high edge density
            if contains_keywords or edge_density > 2000000:  # Adjust threshold as needed
                diagram_frames.append(frame)
                print(f"Frame {idx} is selected as a diagram-related frame.")
            else:
                print(f"Frame {idx} is skipped.")
        except Exception as e:
            print(f"Error processing frame {idx}: {e}")
    return diagram_frames

# Function to extract frames from a video
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
        else:
            print(f"Failed to read frame at index {frame_index}")
    video.release()
    return frames

# Main code execution
if __name__ == "__main__":
    video_path = "data/videos/videoss.mp4"
    keywords = ["architecture"]

    try:
        # Extract frames
        print("Extracting frames...")
        frames = extract_frames(video_path, num_frames=20)

        # Detect diagram-like frames
        print("Detecting diagram frames...")
        diagram_frames = detect_diagram_frames(frames, keywords)

        # Save detected frames
        output_dir = "./"
        for idx, frame in enumerate(diagram_frames):
            frame_path = f"{output_dir}/diagram_{idx}.jpg"
            cv2.imwrite(frame_path, frame)
            print(f"Saved diagram frame to {frame_path}")

    except FileNotFoundError as fnf_error:
        print(f"Tesseract not found or misconfigured: {fnf_error}")
        print("Please ensure Tesseract is installed and added to your PATH.")
    except Exception as e:
        print(f"An error occurred: {e}")
