# AI Healthcare Chatbot - Complete Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Technical Components](#technical-components)
4. [Implementation Details](#implementation-details)
5. [Features](#features)
6. [API Documentation](#api-documentation)
7. [Setup and Installation](#setup-and-installation)
8. [Interview Questions](#interview-questions)

## Project Overview
The AI Healthcare Chatbot is an intelligent medical assistant that uses advanced natural language processing and machine learning to help users assess their health conditions. It combines PubMedBERT's medical knowledge with interactive features and location-based services.

### Key Features
- Symptom analysis and disease prediction
- Interactive follow-up questions
- Nearby hospital recommendations
- Voice input support
- Dark/Light mode
- Session management
- Location services

## Architecture

### Backend Components
1. **Flask Server (app.py)**
   - Handles HTTP requests
   - Manages sessions
   - Integrates with Google Maps API
   - Routes: `/start_session`, `/predict`, `/followup`

2. **BioBERT Helper (biobert_helper.py)**
   - Loads and manages PubMedBERT model
   - Processes medical text
   - Extracts symptoms
   - Generates predictions

3. **Model Components**
   - Fine-tuned PubMedBERT model
   - Symptom-disease mappings
   - Medical terminology processing

### Frontend Components (React)
1. **Main Application (App.jsx)**
   - Chat interface
   - User input handling
   - Location services
   - Theme management

2. **UI Components**
   - Message display
   - Input controls
   - Hospital map
   - Settings panel

## Technical Components

### 1. PubMedBERT Integration
- Pre-trained on medical literature
- Fine-tuned for symptom analysis
- Confidence scoring system
- Medical terminology understanding

### 2. Session Management
```python
@app.route('/start_session', methods=['POST'])
def start_session():
    session_id = str(uuid.uuid4())
    return jsonify({'session_id': session_id})
```

### 3. Disease Prediction
- Symptom extraction
- Context maintenance
- Confidence thresholds
- Follow-up generation

### 4. Location Services
- Google Maps API integration
- Nearby hospital search
- Distance calculation
- Address and location details

## Implementation Details

### Backend API Endpoints

1. **Start Session (`/start_session`)**
   - Creates unique session ID
   - Initializes conversation state
   - Returns session identifier

2. **Predict Disease (`/predict`)**
   - Accepts symptoms and location
   - Returns:
     - Disease predictions
     - Confidence scores
     - Extracted symptoms
     - Nearby hospitals
     - User intent and sentiment

3. **Follow-up Questions (`/followup`)**
   - Manages conversation flow
   - Tracks confirmed/denied symptoms
   - Generates relevant questions

### Frontend Features

1. **Chat Interface**
   - Real-time updates
   - Message history
   - Loading states
   - Error handling

2. **User Input**
   - Text input
   - Voice recognition
   - Symptom confirmation
   - Location sharing

3. **UI/UX**
   - Responsive design
   - Theme switching
   - Smooth animations
   - Accessibility features

## Features

### 1. Medical Analysis
- Symptom recognition
- Disease prediction
- Confidence scoring
- Medical terminology mapping

### 2. Interactive Features
- Follow-up questions
- Voice input
- Location sharing
- Hospital recommendations

### 3. User Experience
- Dark/Light themes
- Message history
- Bookmarking
- Error handling

## API Documentation

### 1. /start_session (POST)
- Creates new chat session
- Returns: `{ session_id: string }`

### 2. /predict (POST)
Request:
```json
{
  "symptoms": string,
  "location": { "lat": number, "lng": number },
  "session_id": string
}
```
Response:
```json
{
  "predictions": Array<{disease: string, confidence: number}>,
  "explanation": string,
  "extracted_symptoms": string[],
  "intent": string,
  "sentiment": string,
  "hospitals": Array<{name: string, address: string, location: {lat: number, lng: number}}>
}
```

### 3. /followup (POST)
Request:
```json
{
  "symptoms": string,
  "denied_symptoms": string[],
  "confirmed_symptoms": string[],
  "session_id": string
}
```
Response:
```json
{
  "question": string,
  "symptom": string
}
```

## Setup and Installation

### Prerequisites
- Python 3.8+
- Node.js 14+
- Google Maps API key

### Backend Setup
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set environment variables:
   ```bash
   export GOOGLE_MAPS_API_KEY="your_api_key"
   ```
3. Run Flask server:
   ```bash
   python app.py
   ```

### Frontend Setup
1. Install dependencies:
   ```bash
   npm install
   ```
2. Run development server:
   ```bash
   npm run dev
   ```

## Interview Questions

### Backend Questions

1. Q: How does the session management work in this application?
   A: The application uses Flask's session management with UUID-based session IDs. Each session tracks conversation state, confirmed/denied symptoms, and prediction history.

2. Q: Explain the role of PubMedBERT in the application.
   A: PubMedBERT is a medical-domain-specific language model that:
   - Understands medical terminology
   - Predicts diseases from symptoms
   - Generates follow-up questions
   - Provides confidence scores

3. Q: How does the system handle multiple symptoms?
   A: The system:
   - Extracts multiple symptoms from text
   - Maintains symptom history
   - Tracks confirmed/denied symptoms
   - Updates predictions accordingly

### Frontend Questions

1. Q: How does the chat interface handle real-time updates?
   A: Using React state management and useEffect hooks for:
   - Message updates
   - Smooth scrolling
   - Loading states
   - Error handling

2. Q: Explain the location feature implementation.
   A: Uses browser's geolocation API and Google Maps API to:
   - Get user location
   - Find nearby hospitals
   - Display results on map
   - Show distance and directions

3. Q: How is the theme switching implemented?
   A: Uses CSS variables and React state to:
   - Toggle between themes
   - Persist user preference
   - Apply consistent styling
   - Smooth transitions

### System Design Questions

1. Q: How would you scale this application?
   A: Several approaches:
   - Load balancing
   - Caching predictions
   - Database optimization
   - Model quantization
   - CDN for static assets

2. Q: How is error handling implemented?
   A: Multiple layers:
   - Frontend validation
   - API error handling
   - Model fallbacks
   - User feedback
   - Logging and monitoring

3. Q: Discuss the security measures in place.
   A: Key features:
   - API key protection
   - Session management
   - CORS configuration
   - Input validation
   - Error sanitization
