# 🎭 Multimodal Emotion Recognition System

An advanced emotion recognition system that integrates both **audio (speech)** and **video (facial expression)** analysis using deep learning models.

## ✨ Features

- 📹 **Real-time Video Emotion Detection** - Facial expression analysis using OpenCV
- 🎤 **Audio Emotion Recognition** - Speech-to-text with emotion classification using DistilBERT
- 🔄 **Multimodal Integration** - Combined audio and video analysis
- 🎨 **Futuristic HUD Interface** - Sci-fi themed web interface with real-time visualization
- 🚀 **Flask REST API** - Backend API for emotion detection services

## 🛠️ Technologies Used

### Backend
- **Python 3.10**
- **Flask** - REST API server
- **OpenCV** - Face detection using Haar Cascade
- **Transformers (HuggingFace)** - DistilBERT for text emotion classification
- **SpeechRecognition** - Audio processing

### Frontend
- **HTML5/CSS3/JavaScript**
- **Tailwind CSS** - Utility-first styling
- **MediaDevices API** - Webcam and microphone access
- **Fetch API** - Backend communication

### ML Models
- **DistilBERT** (`bhadresh-savani/distilbert-base-uncased-emotion`) - Audio emotion classification
- **OpenCV Haar Cascade** - Face detection
- **Neural Networks** - Custom MLP classifier for audio features

## 📋 Prerequisites

- Python 3.10 or higher
- Webcam (for video emotion detection)
- Microphone (for audio emotion detection)
- Modern web browser (Chrome, Firefox, Edge)

## 🚀 Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd "Multimodal Emotion Recognition Integrating Audio and Video Analysis"
```

2. **Create virtual environment**
```bash
python -m venv env_emotion
```

3. **Activate virtual environment**
```bash
# Windows
.\env_emotion\Scripts\Activate.ps1

# Linux/Mac
source env_emotion/bin/activate
```

4. **Install dependencies**
```bash
pip install flask flask-cors opencv-python transformers torch speechrecognition pillow numpy
```

## 💻 Usage

### Running the Backend Server

```bash
python app.py
```

The Flask server will start on `http://localhost:5000`

### Running the Frontend

1. Open `multimodal-analyzer.html` in your web browser
2. Allow camera and microphone permissions when prompted
3. The interface will automatically connect to the backend API

### Available API Endpoints

- `GET /` - Server status and available endpoints
- `GET /api/status` - Health check
- `POST /api/detect/video` - Video emotion detection (accepts base64 image)
- `POST /api/detect/audio` - Audio/text emotion detection (accepts text)
- `POST /api/detect/multimodal` - Combined multimodal detection

## 📁 Project Structure

```
├── app.py                          # Flask backend server
├── multimodal-analyzer.html        # Futuristic HUD frontend
├── combo14.ipynb                   # Working multimodal notebook
├── improvement1.ipynb              # Audio emotion recognition
├── speech and face1.ipynb          # Combined speech and face detection
├── live detection audio.ipynb      # Real-time audio detection
├── MADE/                           # Additional modules
│   └── EMOTION USING VOICE.py
└── output-screenshot/              # Sample outputs
```

## 📸 Screenshots

### Application Interface
![Screenshot 1](output-screenshot/Screenshot%202025-11-25%20192811.png)

### Emotion Detection Demo
![Screenshot 2](output-screenshot/Screenshot%202025-11-25%20192840.png)

### Results View
![Screenshot 3](output-screenshot/Screenshot%202025-11-25%20192905.png)

## 🎯 Emotion Categories

The system can detect the following emotions:

### Video Emotions
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

### Audio Emotions (DistilBERT)
- Joy
- Sadness
- Anger
- Fear
- Love
- Surprise

## 🔧 Configuration

Edit `app.py` to modify:
- API port (default: 5000)
- Model paths
- Confidence thresholds
- Detection intervals

Edit `multimodal-analyzer.html` to modify:
- Frontend polling interval
- UI themes and colors
- Canvas dimensions

## 📊 Model Performance

- **Audio Emotion Recognition**: ~46.67% accuracy (MLP Classifier)
- **Video Face Detection**: Haar Cascade (real-time)
- **Text Emotion Classification**: DistilBERT (pre-trained)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 🎓 Academic Context

This project was developed for emotion recognition research, integrating multiple modalities for improved accuracy and robustness.

## 🐛 Known Issues

- FER library may have installation issues on Windows (permission errors with cv2.pyd)
- Transformers library requires significant disk space for model downloads
- Real-time processing may require GPU for optimal performance

## 🔮 Future Improvements

- [ ] Add deep learning model for video emotion recognition (CNN)
- [ ] Implement emotion fusion algorithms for multimodal integration
- [ ] Add emotion history and analytics dashboard
- [ ] Support for recorded video/audio file analysis
- [ ] Deploy as containerized application (Docker)
- [ ] Add user authentication and session management

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with ❤️ using Python, Flask, and Transformers**
