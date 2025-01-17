# AI-Powered Video Lecture Assistant

A comprehensive web application that transforms video lectures into accessible learning materials through AI-powered transcription, summarization, and visual content extraction.

## 🌟 Features

- **Video Processing**
  - Accept video uploads or URL inputs
  - Support for multiple video formats
  - Efficient processing of long-form educational content

- **AI-Powered Analysis**
  - Multilingual transcription using Whisper AI
  - Intelligent summarization using Facebook BART model
  - Multi-language summary translation (English, Hindi, Marathi)
  - Automatic extraction of important visual frames (diagrams, formulas, etc.)

## 🏗️ Architecture

![Architecture Diagram](https://raw.githubusercontent.com/pratikdhame/Transcribe-Summarize-Visualize/refs/heads/main/architecture.png)

## 🛠️ Technology Stack

### Backend
- FastAPI (Python)
- Whisper AI (OpenAI) - Multilingual transcription
- Facebook BART - Text summarization
- Googletrans - Translation services
- OpenCV - Frame extraction

### Frontend
- React.js

## 🚀 Getting Started

### Prerequisites
```bash
# Python 3.8+
# Node.js 14+
# FFmpeg (for video processing)
```

### Installation

1. Clone the repository
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Setup Backend
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

3. Setup Frontend
```bash
cd transcription-app
npm install
```

### Running the Application

1. Start the Backend Server
```bash
cd backend
python app.py
```

2. Start the Frontend Development Server
```bash
cd transcription-app
npm start
```

## 💡 Usage

1. Access the web application through your browser
2. Either:
   - Upload a video file directly
   - Provide a URL to the video
3. Wait for processing
4. Access:
   - Complete transcription
   - AI-generated summary
   - Translated summaries
   - Extracted important frames

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👥 Authors

- Pratik Dhame
- Soham Borkar

## 🙏 Acknowledgments

- OpenAI for Whisper AI
- Facebook AI Research for BART model
- Google Translate API
