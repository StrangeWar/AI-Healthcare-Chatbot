import React, { useState } from 'react';
import './App.css';

const BACKEND_URL = 'http://localhost:5000/predict';
const FOLLOWUP_URL = 'http://localhost:5000/followup';
const GOOGLE_MAPS_API_KEY = import.meta.env.VITE_REACT_APP_GOOGLE_MAPS_API_KEY;

function App() {
  const [messages, setMessages] = useState([
    { sender: 'bot', text: 'Hi! Please describe your symptoms and location (optional) to get started.' }
  ]);
  const [input, setInput] = useState('');
  const [location, setLocation] = useState(null);
  const [hospitals, setHospitals] = useState([]);
  const [loading, setLoading] = useState(false);
  const [confirmedSymptoms, setConfirmedSymptoms] = useState([]);
  const [deniedSymptoms, setDeniedSymptoms] = useState([]);
  const [lastAskedSymptom, setLastAskedSymptom] = useState(null);

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

  // Handle user message submit
  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
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
      const res = await fetch(BACKEND_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symptoms: input, location })
      });
      const data = await res.json();
      if (data.error) {
        setMessages(msgs => [...msgs, { sender: 'bot', text: data.error }]);
      } else {
        // Only show the backend's explanation (no repetition)
        let reply = data.explanation;
        // Add extracted symptoms and intent to the reply
        if (data.extracted && data.extracted.length > 0) {
          reply += `\n\nI noticed you mentioned these symptoms: ${data.extracted.join(', ')}.`;
        }
        if (data.intent) {
          reply += `\n(Detected intent: ${data.intent})`;
        }
        // If top prediction is above 90%, stop and provide the predicted disease only
        const topPrediction = data.predictions && data.predictions[0];
        if (topPrediction && topPrediction.confidence >= 0.9) {
          reply = `Based on your symptoms, the most likely condition is ${topPrediction.disease} (confidence: ${(topPrediction.confidence * 100).toFixed(1)}%). Please consult a healthcare professional for confirmation.`;
        }
        setMessages(msgs => [...msgs, { sender: 'bot', text: reply }]);
        setHospitals(data.hospitals || []);
        // Fetch follow-up question only if top prediction is below 90%
        if (!(topPrediction && topPrediction.confidence >= 0.9)) {
          const followupRes = await fetch(FOLLOWUP_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              symptoms: input,
              denied_symptoms: newDenied,
              confirmed_symptoms: newConfirmed
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
            }
            setMessages(msgs => [...msgs, { sender: 'bot', text: followupText }]);
            setLastAskedSymptom(followupData.symptom);
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

  return (
    <div className="chatbot-container">
      <div className="chat-header">AI Healthcare Chatbot</div>
      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`chat-msg ${msg.sender}`}>
            <div className={`avatar ${msg.sender}`}>{msg.sender === 'user' ? 'U' : 'B'}</div>
            <div className={`bubble ${msg.sender}`}>{msg.text}</div>
          </div>
        ))}
      </div>
      <form className="chat-input-row" onSubmit={handleSend}>
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Describe your symptoms..."
          disabled={loading}
        />
        <button type="submit" disabled={loading}>Send</button>
        <button type="button" onClick={getLocation} title="Share location">📍</button>
      </form>
      {/* Only show hospitals section if API key is present and hospitals are available */}
      {GOOGLE_MAPS_API_KEY && hospitals.length > 0 && (
        <div className="hospitals-section">
          <h4>Nearby Hospitals</h4>
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
              <li key={idx}><b>{h.name}</b><br />{h.address}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
