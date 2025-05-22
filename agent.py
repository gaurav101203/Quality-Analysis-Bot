import os
import torch
import librosa
import speech_recognition as sr
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import pyttsx3
from flask import Flask, jsonify
from flask_cors import CORS
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from google.cloud import speech, language_v1, aiplatform as vertexai
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
import threading
import time
import requests

# Flask App
app = Flask(__name__)
CORS(app)

# Set Google Credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./rag-model-448019-fe37e29d6e38.json"

credentials = service_account.Credentials.from_service_account_file(
    "./rag-model-448019-fe37e29d6e38.json",
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
credentials.refresh(Request())

# Initialize Google Cloud
vertexai.init(
    project="rag-model-448019",
    location="us-central1",
    credentials=credentials
)

# Load AI Chat Model (Gemini)
model = init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai")

# Load Speech-to-Text Client
speech_client = speech.SpeechClient()

# Load Google Cloud Natural Language API Client
nlp_client = language_v1.LanguageServiceClient()

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()

# Audio Recording Config
fs = 44100  # Sample rate
seconds = 5  # Duration of recording

# Global Data Store
data_store = {"human_text": "", "ai_text": "", "emotion": ""}

# Add a global variable to track listening state
listening_state = {"is_listening": False}

@app.route('/toggle_listening', methods=['POST'])
def toggle_listening():
    """API endpoint to start or stop listening."""
    global listening_state
    listening_state["is_listening"] = not listening_state["is_listening"]
    return jsonify({"is_listening": listening_state["is_listening"]})

@app.route('/get_listening_state', methods=['GET'])
def get_listening_state():
    """API endpoint to fetch the current listening state."""
    return jsonify(listening_state)

@app.route('/update_cuss_words', methods=['POST'])
def update_cuss_words():
    """API endpoint to update cuss words detected in the client."""
    global data_store
    cuss_words = requests.json.get("cuss_words", [])
    data_store["cuss_words"] = cuss_words
    return jsonify({"status": "success"})

@app.route('/get_text', methods=['GET'])
def get_text():
    """API endpoint to fetch transcribed text, AI response, emotion, and cuss words."""
    return jsonify(data_store)

def save_transcript(human_text, ai_text):
    """Save conversation transcript to a text file."""
    with open("conversation.txt", "a", encoding="utf-8") as file:
        file.write(f"Human: {human_text}\n")
        file.write(f"AI: {ai_text}\n")
        file.write("-" * 50 + "\n")

def speak_text(text):
    """Convert text to speech."""
    engine.say(text)
    engine.runAndWait()

def record_audio(filename="audio.wav"):
    """Record audio from the microphone."""
    print("\n Recording...")
    audio_data = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    write(filename, fs, audio_data)

def transcribe_audio(filename="audio.wav"):
    """Convert speech to text using Google Speech-to-Text API."""
    with open(filename, "rb") as audio_file:
        content = audio_file.read()
    
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=fs,
        language_code="en-US",
    )

    response = speech_client.recognize(config=config, audio=audio)
    for result in response.results:
        return result.alternatives[0].transcript
    return ""

def analyze_emotion(text):
    """Analyze emotion from text using Google Cloud Natural Language API."""
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    sentiment = nlp_client.analyze_sentiment(request={"document": document}).document_sentiment
    
    score = sentiment.score
    if score >= 0.3:
        return "Happy"
    elif score <= -0.3:
        return "Angry"
    else:
        return "Neutral"

def capture_and_process_audio():
    """Main function to capture, transcribe, analyze emotion, and respond."""
    global data_store, listening_state
    print("🎤 Say something... (Say 'exit' to stop)")

    while True:
        if not listening_state["is_listening"]:
            time.sleep(1)  # Sleep if not listening
            continue

        try:
            record_audio()
            text = transcribe_audio()
            print(f"📝 Recognized Text: {text}")
            data_store["human_text"] = text

            if text.lower() == "exit":
                print("👋 Exiting...")
                speak_text("Goodbye!")
                break  

            emotion = analyze_emotion(text)
            print(f"🎭 Detected Emotion: {emotion}")
            data_store["emotion"] = emotion

            response = model.invoke([HumanMessage(content=f"{text} (Emotion: {emotion})")])
            response_text = response.content if isinstance(response.content, str) else response.content[0]

            print("🤖 AI Response:", response_text)
            data_store["ai_text"] = response_text  

            save_transcript(text, response_text)  # Save the conversation to a text file
            
            speak_text(response_text)

            # Print cuss words if they exist
            cuss_words = data_store.get("cuss_words", [])
            if cuss_words:
                print(f"🚨 Detected Cuss Words: {', '.join(cuss_words)}")
            else:
                print("✅ No cuss words detected.")

            # Stop listening after processing the response
            listening_state["is_listening"] = False

        except sr.UnknownValueError:
            print("❌ Could not understand the audio")
        except sr.RequestError:
            print("❌ Error connecting to Google Speech API")
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            speak_text("Goodbye!")
            break  
        except Exception as e:
            print(f"⚠️ Error: {e}")

        time.sleep(1)

if __name__ == '__main__':
    # Start Flask in a separate thread
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False), daemon=True).start()

    # Start voice recognition
    capture_and_process_audio()
