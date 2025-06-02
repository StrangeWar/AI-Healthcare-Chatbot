from flask import Flask, request, jsonify, session
from flask_cors import CORS
from biobert_helper import predict_disease, get_followup_question
import os
import requests
import uuid

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session handling
CORS(app, supports_credentials=True)

# Google Maps API Key (set as environment variable for security)
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')
GOOGLE_MAPS_SEARCH_URL = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json'

@app.route('/start_session', methods=['POST'])
def start_session():
    session_id = str(uuid.uuid4())
    return jsonify({'session_id': session_id})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = data.get('symptoms', '')
    location = data.get('location', None)
    session_id = data.get('session_id')
    
    if not symptoms:
        return jsonify({'error': 'No symptoms provided.'}), 400
    
    if not session_id:
        return jsonify({'error': 'No session ID provided.'}), 400
    
    # Predict disease using BioBERT
    try:
        predictions, explanation, extracted, intent, sentiment = predict_disease(symptoms, session_id)
    except Exception as e:
        import traceback
        return jsonify({
            'error': 'Prediction failed.',
            'details': str(e),
            'trace': traceback.format_exc()
        }), 500

    # Find nearby hospitals if location is provided
    hospitals = []
    if location and GOOGLE_MAPS_API_KEY:
        try:
            params = {
                'location': f"{location['lat']},{location['lng']}",
                'radius': 5000,
                'type': 'hospital',
                'key': GOOGLE_MAPS_API_KEY
            }
            resp = requests.get(GOOGLE_MAPS_SEARCH_URL, params=params)
            results = resp.json().get('results', [])
            hospitals = [{
                'name': h['name'],
                'address': h.get('vicinity'),
                'location': h['geometry']['location']
            } for h in results[:5]]
        except Exception as e:
            hospitals = []
    
    return jsonify({
        'predictions': predictions,
        'explanation': explanation,
        'extracted_symptoms': extracted,
        'intent': intent,
        'sentiment': sentiment,
        'hospitals': hospitals
    })

@app.route('/followup', methods=['POST'])
def followup():
    data = request.get_json()
    symptoms = data.get('symptoms', '')
    denied_symptoms = data.get('denied_symptoms', [])
    confirmed_symptoms = data.get('confirmed_symptoms', [])
    session_id = data.get('session_id')
    
    if not symptoms:
        return jsonify({'error': 'No symptoms provided.'}), 400
    
    if not session_id:
        return jsonify({'error': 'No session ID provided.'}), 400
        
    try:
        question, symptom = get_followup_question(
            symptoms,
            session_id,
            denied_symptoms,
            confirmed_symptoms
        )
        return jsonify({
            'question': question,
            'symptom': symptom
        })
    except Exception as e:
        import traceback
        return jsonify({
            'error': 'Failed to generate follow-up question.',
            'details': str(e),
            'trace': traceback.format_exc()
        }), 500

@app.route('/')
def home():
    return 'Healthcare Chatbot Backend is running.'

if __name__ == '__main__':
    app.run(debug=True)
