from flask import Flask, request, jsonify
import speech_recognition as sr
from transformers import pipeline
import wave
import os

app = Flask(__name__)

def recognize_speech_from_file(audio_file):
    recognizer = sr.Recognizer()
    
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)


    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.RequestError:
        return "Could not request results from Google Speech Recognition service."
    except sr.UnknownValueError:
        return "Could not understand audio."

def detect_emotion(text):
    emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
    emotions = emotion_classifier(text)
    return emotions

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    audio_path = 'uploaded_audio.wav'
    file.save(audio_path)

    spoken_text = recognize_speech_from_file(audio_path)
    
    if spoken_text:
        emotions = detect_emotion(spoken_text)
        
        return jsonify({"spoken_text": spoken_text, "emotions": emotions})

    return jsonify({"error": "Could not process audio"}), 500

if __name__ == "__main__":
    app.run(debug=True)