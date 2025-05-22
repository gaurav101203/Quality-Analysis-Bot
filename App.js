import React, { useState, useEffect } from "react";
import axios from "axios";
import "./App.css"; // Import the CSS file
import { Mic, Loader2 } from "lucide-react";

const App = () => {
  const [humanText, setHumanText] = useState(""); // Human's speech
  const [aiText, setAiText] = useState(""); // AI's response
  const [loading, setLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch the human text and AI response
        const { data } = await axios.get("http://localhost:5000/get_text");
        setHumanText(data.human_text);
        setAiText(data.ai_text);

        // Fetch the listening state from the server
        const listeningState = await axios.get("http://localhost:5000/get_listening_state");
        setIsListening(listeningState.data.is_listening);
        if (!listeningState.data.is_listening) {
          setLoading(false); // Reset button state
        }
      } catch (error) {
        console.error("Error fetching data:", error);
      }
    };

    const interval = setInterval(fetchData, 2000); // Fetch data every 2 seconds
    return () => clearInterval(interval);
  }, []);

  const toggleListening = async () => {
    setLoading(true); // Set loading to true when button is clicked
    try {
      const response = await axios.post("http://localhost:5000/toggle_listening");
      setIsListening(response.data.is_listening);
    } catch (error) {
      console.error("Error toggling listening state:", error);
    }
  };

  return (
    <div className="container">
      <div className="output">
        <h1>IA Bot</h1>
        <div className="chat-box">
          <div className="chat-box-human">
            <strong>🙍‍♂️ You:</strong> {humanText || "Say something..."}
          </div>
          <div className="chat-box-ai">
            <strong>🤖 AI:</strong> {aiText || "Waiting for response..."}
          </div>
        </div>
      </div>

      {/* Button positioned at the bottom-center */}
      <div
        className="mic-container"
        onClick={toggleListening}
        disabled={loading || isListening}
      >
        {loading ? (
          <Loader2 className="loading mic-icon" />
        ) : (
          <Mic className="mic-icon" />
        )}
      </div>


      {isListening && <p className="listening-status">🎤 The server is currently listening...</p>}
    </div>
  );
};

export default App;
