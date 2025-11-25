import speech_recognition as sr
from transformers import pipeline

print("🎤 Speech Emotion Recognition Test")
print("=" * 50)

test_texts = [
    "I am so happy today!",
    "This is terrible and I'm very angry",
    "I'm feeling scared and afraid",
    "What a wonderful surprise!",
    "I feel calm and peaceful"
]

print("\n📦 Loading emotion detection model...")
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
print("✓ Model loaded successfully!\n")

print("🔮 Testing emotion detection on sample texts:\n")
for text in test_texts:
    print(f"Text: '{text}'")
    emotions = emotion_classifier(text)
    for emotion in emotions:
        print(f"  → Emotion: {emotion['label']}, Confidence: {emotion['score']:.2%}")
    print()

print("=" * 50)
print("✨ Test complete!")
print("\nTo use with microphone, run: python app.py")
