import os
import requests
from transformers import pipeline
from better_profanity import profanity
import re
import time

# Flask server URL (replace with the actual IP of the machine running the server)
FLASK_SERVER_URL = "http://192.168.20.176:5000/get_text"  # Update with the correct IP

# Load sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

# Function to detect slang words
def detect_slang(text):
    slang_list = ["gonna", "wanna", "lemme", "y'all", "ain't", "bruh", "dope", "lit", "sus"]  # Add more as needed
    words = re.findall(r'\b\w+\b', text)  # Extract words without punctuation
    return [word for word in words if word in slang_list]

# Function to detect cuss words and profanity probability
def analyze_profanity(text):
    profanity.load_censor_words()  # Ensure the profanity word list is loaded
    print(f"Analyzing text for profanity: {text}")  # Log the text being analyzed

    cuss_words = [word for word in text.split() if profanity.contains_profanity(word)]
    print(f"Cuss words detected: {cuss_words}")  # Log detected cuss words

    profanity_probability = len(cuss_words) / max(len(text.split()), 1)  # Normalize by total words
    print(f"Profanity probability: {profanity_probability}")  # Log profanity probability

    return cuss_words, round(profanity_probability, 2)
# Function to analyze text
def analyze_text(text, label):
    if not text.strip():
        return f"{label}: No text found."

    sentiment = sentiment_analyzer(text)[0]
    slang_words = detect_slang(text)
    cuss_words, profanity_probability = analyze_profanity(text)

    return {
        "Label": label,
        "Sentiment": sentiment,
        "Slang Words": slang_words,
        "Cuss Words": cuss_words,
        "Profanity Probability": profanity_probability
    }

# Main function to fetch and analyze data
def main():
    while True:
        try:
            # Fetch data from Flask server
            response = requests.get(FLASK_SERVER_URL)
            if response.status_code == 200:
                data = response.json()
                print("✅ Data fetched successfully from the server.")

                # Check if required data is present
                if "human_text" not in data or "ai_text" not in data:
                    print("❌ Missing 'human_text' or 'ai_text' in the response.")
                    continue

                human_text = data.get("human_text", "")
                ai_text = data.get("ai_text", "")

                # Analyze human and AI text
                human_analysis = analyze_text(human_text, "Human Text")
                ai_analysis = analyze_text(ai_text, "AI Text")

                print("\n🔍 Sentiment Analysis Results:")
                print(human_analysis)
                print(ai_analysis)

                # Ensure human_analysis is a dictionary and has "Cuss Words"
                if isinstance(human_analysis, dict):
                    cuss_words = human_analysis.get("Cuss Words", [])
                    if cuss_words:
                        # Send cuss words to the Flask server
                        update_response = requests.post(
                            "http://192.168.20.176:5000/update_cuss_words",  # Update with the correct IP
                            json={"cuss_words": cuss_words}
                        )
                        if update_response.status_code == 200:
                            print(f"✅ Sent cuss words to server: {cuss_words}")
                    else:
                        print("No cuss words detected.")
            else:
                print(f"❌ Failed to fetch data from server. HTTP Status Code: {response.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"❌ Error connecting to the Flask server: {str(e)}")
        
        # Wait for a few seconds before fetching data again (adjust the time as needed)
        time.sleep(10)

if __name__ == "__main__":
    main()
