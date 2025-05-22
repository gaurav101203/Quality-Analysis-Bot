import os
import requests
from transformers import pipeline
from better_profanity import profanity
import re
import time
import tkinter as tk
from tkinter import filedialog
from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Flask server URL (replace with the actual IP of the machine running the server)
FLASK_SERVER_URL = "http://192.168.20.176:5000/convert"  # Update with the correct IP

# Load sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

# Load BERT model for similarity
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to calculate accuracy using cosine similarity
def calculate_accuracy(ai_response, ground_truth):
    embeddings = similarity_model.encode([ai_response, ground_truth], convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return round(float(similarity_score), 4)  # Convert tensor to float

# Function to detect slang words
def detect_slang(text):
    slang_list = ["gonna", "wanna", "lemme", "y'all", "ain't", "bruh", "dope", "lit", "sus"]  # Add more as needed
    words = re.findall(r'\b\w+\b', text)  # Extract words without punctuation
    return [word for word in words if word in slang_list]

# Function to detect cuss words and profanity probability
def analyze_profanity(text):
    profanity.load_censor_words()
    cuss_words = [word for word in text.split() if profanity.contains_profanity(word)]
    profanity_probability = len(cuss_words) / max(len(text.split()), 1)  # Normalize by total words
    return cuss_words, round(profanity_probability, 2)

# Function to analyze text
def analyze_text(text, label):
    # if not text.strip():
    #     return f"{label}: No text found."

    sentiment = sentiment_analyzer(text)[0]
    slang_words = detect_slang(text)
    cuss_words, profanity_probability = analyze_profanity(text)

    return {
        "Sentiment": sentiment,
        "Slang Words": slang_words,
        "Cuss Words": cuss_words,
        "Profanity Probability": profanity_probability
    }

# Function to send MP3 to the Flask server and get the analysis
def send_mp3_to_server(mp3_file_path):
    files = {'file': open(mp3_file_path, 'rb')}
    try:
        response = requests.post(FLASK_SERVER_URL, files=files)
        if response.status_code == 200:
            result = response.json()
            transcribed_text = result.get("transcribed_text", "")
            return transcribed_text
        else:
            print("Failed to get transcription:", response.status_code)
            return ""
    except requests.exceptions.RequestException as e:
        print("Error connecting to the Flask server:", str(e))
        return ""

# Function to load the ground truth from a CSV file
def load_ground_truth(csv_file):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Check if the columns exist
        if "questions" not in df.columns or "answers" not in df.columns:
            print("Error: CSV file must contain 'Question' and 'Ground Truth' columns.")
            return None
        
        # Convert the DataFrame to a dictionary (optional, for easy look-up)
        ground_truth_dict = dict(zip(df['questions'].str.lower().str.strip(), df['answers'].str.strip()))
        return ground_truth_dict
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

# Function to get the ground truth based on the transcription (e.g., question)
def get_ground_truth_from_csv(csv_file, transcribed_text):
    ground_truth_dict = load_ground_truth(csv_file)
    
    if ground_truth_dict is None:
        return None  # If CSV loading failed, return None

    # Find the closest matching question in the ground truth CSV
    transcribed_text_lower = transcribed_text.lower().strip()
    best_match = None
    highest_score = 0

    for question, answer in ground_truth_dict.items():
        # Use cosine similarity (or simple text matching) to find the best match
        score = calculate_accuracy(transcribed_text_lower, question)
        
        if score > highest_score:
            highest_score = score
            best_match = answer

    return best_match if best_match else "No match found"

# Function to allow manual file selection
def select_mp3_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select an MP3 file", filetypes=[("MP3 Files", "*.mp3")])
    return file_path

# Main function to fetch and analyze data
def main():
    # Specify the CSV file containing the ground truth
    ground_truth_csv = "your_questions_answers.csv"
    
    while True:
        # Allow the user to manually select the MP3 file
        mp3_file_path = select_mp3_file()

        if mp3_file_path:
            print(f"Selected file: {mp3_file_path}")
            
            # Send MP3 and get transcribed text from Flask server
            transcribed_text = send_mp3_to_server(mp3_file_path)
            
            if transcribed_text:  # Ensure we have a valid response from the Flask server
                print(f"Transcribed Text: {transcribed_text}")
                
                # Retrieve the best matching ground truth from CSV
                ground_truth = get_ground_truth_from_csv(ground_truth_csv, transcribed_text)

                # Calculate accuracy and analyze sentiment
                if ground_truth:
                    print(f"\nGround Truth: {ground_truth}")
                    accuracy_score = calculate_accuracy(transcribed_text, ground_truth)
                    sentiment_result = analyze_text(transcribed_text, "Transcribed Text")

                    print("\nSentiment Analysis Results for Transcribed Text:")
                    print(sentiment_result)

                    # Print the accuracy score
                    print(f"\nAccuracy Score: {accuracy_score}")
                else:
                    print("No matching ground truth found for the transcription.")
        else:
            print("No file selected, please try again.")
        
        time.sleep(10)  # Sleep for 10 seconds before making another request

if __name__ == '__main__':
    main()
