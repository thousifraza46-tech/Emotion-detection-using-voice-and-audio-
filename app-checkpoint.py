import speech_recognition as sr
from transformers import pipeline

def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Please speak something...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.RequestError:
        print("Could not request results from Google Speech Recognition service.")
        return None
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None

def detect_emotion(text):
    emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

    emotions = emotion_classifier(text)
    return emotions

def main():
    spoken_text = recognize_speech_from_mic()
    
    if spoken_text:
        emotions = detect_emotion(spoken_text)
        
        for emotion in emotions:
            print(f"Emotion: {emotion['label']}, Score: {emotion['score']:.4f}")

if _name_ == "_main_":
    main()




