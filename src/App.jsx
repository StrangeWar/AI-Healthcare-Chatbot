import React, { useState, useEffect, useRef } from 'react';
import './App.css';

// Add SpeechRecognition setup outside component to avoid re-initialization
let speechRecognition = null;
try {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SpeechRecognition) {
    speechRecognition = new SpeechRecognition();
    speechRecognition.continuous = false;
    speechRecognition.interimResults = true;
    speechRecognition.lang = 'en-US';
  }
} catch (error) {
  console.error('Speech Recognition is not supported:', error);
}

const BACKEND_URL = 'http://localhost:5000';
const GOOGLE_MAPS_API_KEY = import.meta.env.VITE_REACT_APP_GOOGLE_MAPS_API_KEY;

function App() {
  const [messages, setMessages] = useState([
    { 
      sender: 'bot', 
      text: 'Hi! 👋 I\'m your AI Health Assistant. Please describe any symptoms you\'re experiencing, and I\'ll help assess your condition. You can also share your location to find nearby hospitals.' 
    }
  ]);
  const [input, setInput] = useState('');
  const [location, setLocation] = useState(null);
  const [hospitals, setHospitals] = useState([]);
  const [loading, setLoading] = useState(false);
  const [confirmedSymptoms, setConfirmedSymptoms] = useState([]);
  const [deniedSymptoms, setDeniedSymptoms] = useState([]);
  const [lastAskedSymptom, setLastAskedSymptom] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [conversations, setConversations] = useState([
    { id: 'current', name: 'Current Chat', timestamp: new Date() }
  ]);
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [bookmarkedMessages, setBookmarkedMessages] = useState([]);
  const [isListening, setIsListening] = useState(false);
  const [extractedSymptoms, setExtractedSymptoms] = useState([]);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Initialize session when component mounts
  useEffect(() => {
    const initSession = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/start_session`, {
          method: 'POST',
        });
        const data = await response.json();
        setSessionId(data.session_id);
      } catch (error) {
        console.error('Failed to initialize session:', error);
      }
    };
    initSession();
  }, []);

  // Get user geolocation
  const getLocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        pos => setLocation({ lat: pos.coords.latitude, lng: pos.coords.longitude }),
        err => alert('Location access denied.')
      );
    } else {
      alert('Geolocation is not supported by this browser.');
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Focus input on load
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Handle input key press
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend(e);
    }
  };

  // Handle user message submit
  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    if (!sessionId) {
      setMessages(msgs => [...msgs, { sender: 'bot', text: 'Please wait while the session initializes...' }]);
      return;
    }
    setMessages([...messages, { sender: 'user', text: input }]);
    setLoading(true);

    // Stop follow-up if user says no more symptoms after 'Can you describe any other symptoms you have?'
    if (
      messages.length > 0 &&
      messages[messages.length - 1].text.includes("Can you describe any other symptoms you have?") &&
      /^(no|none|nothing|i don't have|no other)/i.test(input.trim())
    ) {
      setMessages(msgs => [
        ...msgs,
        { sender: 'bot', text: "Thank you for sharing. Based on what you've told me, I recommend consulting a healthcare professional for a more accurate diagnosis. If you notice any new symptoms, feel free to let me know!" }
      ]);
      setInput('');
      setLoading(false);
      return;
    }

    // If lastAskedSymptom is set, treat this input as an answer to the follow-up question
    let newConfirmed = [...confirmedSymptoms];
    let newDenied = [...deniedSymptoms];
    if (lastAskedSymptom) {
      if (/\byes?\b/i.test(input)) {
        newConfirmed.push(lastAskedSymptom);
      } else if (/\bno?\b/i.test(input)) {
        newDenied.push(lastAskedSymptom);
      }
    }

    try {
      // On first message, do disease prediction and ask follow-up
      const res = await fetch(`${BACKEND_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          symptoms: input, 
          location,
          session_id: sessionId 
        })
      });
      const data = await res.json();
      if (data.error) {
        setMessages(msgs => [...msgs, { sender: 'bot', text: data.error }]);
      } else {
        // Only show the backend's explanation (no repetition)
        let reply = data.explanation;
        // Add extracted symptoms and intent to the reply
        if (data.extracted_symptoms && data.extracted_symptoms.length > 0) {
          reply += `\n\nI noticed you mentioned these symptoms: ${data.extracted_symptoms.join(', ')}.`;
        }
        // Don't show intent in user messages
        // Now always proceed to follow-up questions unless confidence is extremely high
        const topPrediction = data.predictions && data.predictions[0];
        if (topPrediction && topPrediction.confidence >= 95) {
          reply = `Based on your symptoms, the most likely condition is ${topPrediction.disease} (confidence: ${topPrediction.confidence.toFixed(1)}%). Please consult a healthcare professional for confirmation.`;
        }
        setMessages(msgs => [...msgs, { sender: 'bot', text: reply }]);
        setHospitals(data.hospitals || []);
        // Always fetch follow-up questions unless we have extremely high confidence
        if (!(topPrediction && topPrediction.confidence >= 95)) {
          const followupRes = await fetch(`${BACKEND_URL}/followup`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              symptoms: input,
              denied_symptoms: newDenied,
              confirmed_symptoms: newConfirmed,
              session_id: sessionId
            })
          });
          const followupData = await followupRes.json();
          if (followupData.error) {
            let errorMsg = followupData.error;
            if (followupData.details) errorMsg += `\nDetails: ${followupData.details}`;
            if (followupData.trace) errorMsg += `\nTraceback: ${followupData.trace}`;
            setMessages(msgs => [...msgs, { sender: 'bot', text: errorMsg }]);
          } else if (followupData.question) {
            // Make follow-up question more conversational
            let followupText = followupData.question;
            if (followupData.symptom) {
              followupText = `Just to help me understand better, ${followupText.toLowerCase()}`;
              setLastAskedSymptom(followupData.symptom);
            } else {
              setLastAskedSymptom(null);
            }
            
            // Only add the follow-up question if it's not the "confident in my assessment" message
            if (followupData.question && !followupData.question.includes("confident in my assessment")) {
              setMessages(msgs => [...msgs, { sender: 'bot', text: followupText }]);
            }
          } else {
            setLastAskedSymptom(null);
          }
        } else {
          setLastAskedSymptom(null);
        }
        setConfirmedSymptoms(newConfirmed);
        setDeniedSymptoms(newDenied);
      }
    } catch (err) {
      setMessages(msgs => [...msgs, { sender: 'bot', text: 'Sorry, something went wrong.' }]);
    }
    setInput('');
    setLoading(false);
  };

  const startNewChat = () => {
    setMessages([{ 
      sender: 'bot', 
      text: 'Hi! 👋 I\'m your AI Health Assistant. Please describe any symptoms you\'re experiencing, and I\'ll help assess your condition. You can also share your location to find nearby hospitals.' 
    }]);
    setConfirmedSymptoms([]);
    setDeniedSymptoms([]);
    setHospitals([]);
    const newChat = {
      id: Date.now().toString(),
      name: 'New Chat',
      timestamp: new Date()
    };
    setConversations([newChat, ...conversations]);
  };

  // Handle voice input with improved error handling and feedback
  const handleVoiceInput = () => {
    if (!speechRecognition) {
      alert('Speech recognition is not supported in your browser. Please try using Chrome, Edge, or Safari.');
      return;
    }

    if (isListening) {
      speechRecognition.stop();
    } else {
      try {
        setInput(''); // Clear existing input
        speechRecognition.start();
        setIsListening(true);
      } catch (error) {
        console.error('Speech recognition error:', error);
        alert('Failed to start voice input. Please try again.');
        setIsListening(false);
      }
    }
  };

  // Set up speech recognition event handlers
  useEffect(() => {
    if (!speechRecognition) return;

    const handleResult = (event) => {
      const transcript = Array.from(event.results)
        .map(result => result[0])
        .map(result => result.transcript)
        .join('');
      
      setInput(transcript);
      
      // If we have a final result, stop listening
      if (event.results[0].isFinal) {
        speechRecognition.stop();
        setIsListening(false);
      }
    };

    const handleError = (event) => {
      console.error('Speech recognition error:', event.error);
      setIsListening(false);
      if (event.error === 'not-allowed') {
        alert('Microphone access was denied. Please allow microphone access to use voice input.');
      } else {
        alert('Voice input error. Please try again.');
      }
    };

    const handleEnd = () => {
      setIsListening(false);
    };

    speechRecognition.addEventListener('result', handleResult);
    speechRecognition.addEventListener('error', handleError);
    speechRecognition.addEventListener('end', handleEnd);

    return () => {
      speechRecognition.removeEventListener('result', handleResult);
      speechRecognition.removeEventListener('error', handleError);
      speechRecognition.removeEventListener('end', handleEnd);
    };
  }, []);

  // Toggle message bookmark
  const toggleBookmark = (messageIndex) => {
    setBookmarkedMessages(prev => {
      if (prev.includes(messageIndex)) {
        return prev.filter(i => i !== messageIndex);
      }
      return [...prev, messageIndex];
    });
  };

  // Export chat history
  const exportChat = () => {
    const chatHistory = messages.map(msg => ({
      ...msg,
      timestamp: new Date().toISOString(),
    }));

    const blob = new Blob([JSON.stringify(chatHistory, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-history-${new Date().toISOString()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className={`app-container ${isDarkMode ? 'dark' : 'light'}`}>
      <div className="sidebar">
        <div className="sidebar-header">
          <button className="new-chat-btn" onClick={startNewChat}>
            <span>+</span>New chat
          </button>
          <button 
            className="theme-toggle-btn" 
            onClick={() => setIsDarkMode(!isDarkMode)}
            title={isDarkMode ? "Switch to light mode" : "Switch to dark mode"}
          >
            {isDarkMode ? '☀️' : '🌙'}
          </button>
        </div>
        <div className="conversations-list">
          <button className="export-btn" onClick={exportChat} title="Export chat history">
            💾 Export Chat
          </button>
          {conversations.map(chat => (
            <div key={chat.id} className="conversation-item">
              <span className="chat-icon">💬</span>
              <span className="chat-name">{chat.name}</span>
              <span className="chat-time">{chat.timestamp.toLocaleTimeString()}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="main-content">
        <div className="chatbot-container">
          <div className="chat-header">
            <img src="/vite.svg" alt="AI Health Assistant" />
            <span>AI Health Assistant</span>
            {extractedSymptoms.length > 0 && (
              <div className="symptom-tags">
                {extractedSymptoms.map((symptom, index) => (
                  <span key={index} className="symptom-tag">
                    {symptom}
                  </span>
                ))}
              </div>
            )}
          </div>
          <div className="chat-messages">
            {messages.map((msg, i) => (
              <div key={i} className={`chat-msg ${msg.sender}`}>
                <div className={`avatar ${msg.sender}`}>
                  {msg.sender === 'user' ? '👤' : '🩺'}
                </div>
                <div className="message-content">
                  <div className={`bubble ${msg.sender}`}>
                    {msg.text}
                    <div className="message-timestamp">
                      {new Date().toLocaleTimeString()}
                    </div>
                  </div>
                  <div className="message-actions">
                    <button 
                      className={`bookmark-btn ${bookmarkedMessages.includes(i) ? 'active' : ''}`}
                      onClick={() => toggleBookmark(i)}
                      title={bookmarkedMessages.includes(i) ? "Remove bookmark" : "Add bookmark"}
                    >
                      {bookmarkedMessages.includes(i) ? '⭐' : '☆'}
                    </button>
                  </div>
                </div>
              </div>
            ))}
            {loading && (
              <div className="chat-msg bot">
                <div className="avatar bot">🩺</div>
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          <form className="chat-input-row" onSubmit={handleSend}>
            <button 
              type="button" 
              onClick={handleVoiceInput}
              className={`voice-input-btn ${isListening ? 'active' : ''}`}
              title={isListening ? "Stop voice input" : "Start voice input"}
            >
              {isListening ? '🎤' : '🎤'}
            </button>
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={isListening ? "Listening..." : "Describe your symptoms..."}
              disabled={loading || !sessionId}
              className={isListening ? 'listening' : ''}
            />
            <button type="submit" disabled={loading || !sessionId || !input.trim()}>
              {loading ? '...' : 'Send'}
            </button>
            <button 
              type="button" 
              onClick={getLocation} 
              title="Share location to find nearby hospitals"
              disabled={loading}
            >
              📍
            </button>
          </form>
          {GOOGLE_MAPS_API_KEY && hospitals.length > 0 && (
            <div className="hospitals-section">
              <h4>📍 Nearby Hospitals</h4>
              <div className="map-container">
                <iframe
                  title="Nearby Hospitals"
                  width="100%"
                  height="250"
                  style={{ border: 0 }}
                  loading="lazy"
                  allowFullScreen
                  src={`https://www.google.com/maps/embed/v1/search?key=${GOOGLE_MAPS_API_KEY}&q=hospital&center=${location?.lat},${location?.lng}&zoom=13`}
                />
              </div>
              <ul>
                {hospitals.map((h, idx) => (
                  <li key={idx}>
                    <strong>{h.name}</strong>
                    <div className="hospital-address">{h.address}</div>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
