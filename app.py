from flask import Flask, request, jsonify
from flask_cors import CORS
from biobert_helper import predict_disease, get_followup_question
import os
import requests

app = Flask(__name__)
CORS(app)

# Google Maps API Key (set as environment variable for security)
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')
GOOGLE_MAPS_SEARCH_URL = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = data.get('symptoms', '')
    location = data.get('location', None)
    if not symptoms:
        return jsonify({'error': 'No symptoms provided.'}), 400
    
    # Predict disease using BioBERT
    try:
        # Unpack all returned values for debugging
        predictions, explanation, extracted, intent, sentiment = predict_disease(symptoms)
    except Exception as e:
        import traceback
        return jsonify({'error': 'Prediction failed.', 'details': str(e), 'trace': traceback.format_exc()}), 500

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
            hospitals = [
                {
                    'name': h['name'],
                    'address': h.get('vicinity'),
                    'location': h['geometry']['location']
                } for h in results[:5]
            ]
        except Exception as e:
            hospitals = []
    
    return jsonify({
        'predictions': predictions,
        'explanation': explanation,
        'hospitals': hospitals
    })

@app.route('/followup', methods=['POST'])
def followup():
    data = request.get_json()
    symptoms = data.get('symptoms', '')
    denied_symptoms = data.get('denied_symptoms', [])
    confirmed_symptoms = data.get('confirmed_symptoms', [])
    if not symptoms:
        return jsonify({'error': 'No symptoms provided.'}), 400
    try:
        question, symptom = get_followup_question(symptoms, denied_symptoms, confirmed_symptoms)
        return jsonify({'question': question, 'symptom': symptom})
    except Exception as e:
        import traceback
        return jsonify({'error': 'Failed to generate follow-up question.', 'details': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/')
def home():
    return 'Healthcare Chatbot Backend is running.'

if __name__ == '__main__':
    app.run(debug=True)
