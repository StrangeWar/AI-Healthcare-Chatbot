import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, pipeline
import torch.nn.functional as F
import re
import pandas as pd
import spacy
from textblob import TextBlob
import json

# Load spaCy model for NER
try:
    nlp_spacy = spacy.load('en_core_web_sm')
except:
    import os
    os.system('python -m spacy download en_core_web_sm')
    nlp_spacy = spacy.load('en_core_web_sm')

# Load Hugging Face pipelines for intent and sentiment
try:
    intent_classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
    sentiment_analyzer = pipeline('sentiment-analysis')
except Exception as e:
    intent_classifier = None
    sentiment_analyzer = None

# Synonym mapping (expand as needed)
SYNONYM_MAP = {
    'tummy ache': 'abdominal pain',
    'stomach ache': 'abdominal pain',
    'runny nose': 'rhinorrhea',
    'sore throat': 'pharyngitis',
    'high temperature': 'fever',
    'throwing up': 'vomiting',
    # ...add more as needed...
}

# Load PubMedBERT model and tokenizer (replace with your fine-tuned model path if needed)
MODEL_PATH = 'pubmedbert-finetuned'
config = AutoConfig.from_pretrained(MODEL_PATH)
print(f"Loaded model config from: {MODEL_PATH}")
# Try to load disease labels from label_order.json for real names
try:
    with open('label_order.json') as f:
        DISEASE_LABELS = json.load(f)
    print(f"Loaded disease labels from label_order.json: {DISEASE_LABELS}")
except Exception:
    # Fallback to config id2label (may be LABEL_x)
    try:
        DISEASE_LABELS = [config.id2label[k] for k in sorted(config.id2label, key=lambda x: int(x) if x.isdigit() else int(x.split('_')[-1]))]
    except Exception:
        DISEASE_LABELS = list(config.id2label.values())
    print(f"Loaded disease labels from config: {DISEASE_LABELS}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
print(f"Model and tokenizer loaded from: {MODEL_PATH}")

# Disease to key symptoms mapping
DISEASE_SYMPTOMS = {
    'Bronchitis': ['fatigue', 'persistent cough with mucus', 'chest discomfort'],
    'Dengue': ['high fever', 'headache', 'rash', 'joint pain', 'muscle pain', 'chills', 'sweats'],
    'Allergic Rhinitis': ['runny nose', 'nasal congestion', 'sneezing', 'itchy eyes'],
    'Sinusitis': ['facial pain', 'runny nose', 'nasal congestion', 'headache'],
    'Pneumonia': ['fever', 'cough', 'chest pain', 'shortness of breath'],
    'Chickenpox': ['fever', 'rash', 'loss of appetite', 'fatigue'],
    'Flu': ['fever', 'fatigue', 'chills', 'muscle pain', 'cough', 'sore throat'],
    'Meningitis': ['high fever', 'headache', 'stiff neck', 'sensitivity to light'],
    'Common Cold': ['mild headache', 'sore throat', 'sneezing', 'runny nose'],
    'Malaria': ['chills', 'high fever', 'headache', 'nausea', 'sweats'],
    'Gastroenteritis': ['abdominal pain', 'diarrhea', 'vomiting', 'nausea', 'fever'],
    'COVID-19': ['loss of taste', 'high fever', 'dry cough', 'fatigue', 'shortness of breath'],
    'Asthma': ['coughing', 'shortness of breath', 'wheezing', 'chest tightness'],
    'Tuberculosis': ['persistent cough', 'weight loss', 'chest pain', 'night sweats'],
    'Migraine': ['headache', 'nausea', 'vomiting', 'light sensitivity', 'dizziness'],
}

def correct_spelling(text):
    blob = TextBlob(text)
    return str(blob.correct())

def map_synonyms(text):
    for k, v in SYNONYM_MAP.items():
        text = text.replace(k, v)
    return text

def extract_symptoms(text):
    doc = nlp_spacy(text)
    symptoms = [ent.text for ent in doc.ents if ent.label_ in ['SYMPTOM', 'DISEASE', 'CONDITION']]
    return symptoms

def detect_intent(text):
    if not intent_classifier:
        return 'symptom_report'
    candidate_labels = ['reporting symptoms', 'asking for advice', 'seeking hospital info']
    result = intent_classifier(text, candidate_labels)
    return result['labels'][0]

def detect_sentiment(text):
    if not sentiment_analyzer:
        return 'neutral'
    result = sentiment_analyzer(text)
    return result[0]['label']

def preprocess_text(text):
    text = correct_spelling(text)
    text = map_synonyms(text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_disease(symptoms_text):
    # Preprocess input
    clean_text = preprocess_text(symptoms_text)
    print(f"Preprocessed input: {clean_text}")
    extracted = extract_symptoms(symptoms_text)
    intent = detect_intent(symptoms_text)
    sentiment = detect_sentiment(symptoms_text)
    model.eval()  # Ensure model is in eval mode
    inputs = tokenizer(clean_text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).squeeze().tolist()
    predictions = [
        {'disease': label, 'confidence': float(round(prob, 3))}
        for label, prob in zip(DISEASE_LABELS, probs)
    ]
    predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    top = predictions[0]
    threshold = 0.2
    if top['confidence'] < threshold:
        explanation = "The model is not confident about the prediction. Please provide more details or consult a healthcare professional."
    else:
        explanation = f"From what you've described, here are some possible conditions: " + \
            ' • '.join([f"{p['disease']} ({p['confidence']}%)" for p in predictions[:5]]) + \
            f" Based on your symptoms, the most likely condition is {top['disease']} (confidence: {top['confidence']}%). Please consult a healthcare professional for confirmation."
    # Return extracted symptoms, intent, and sentiment for frontend display
    return predictions, explanation, extracted, intent, sentiment

def get_followup_question(user_text, denied_symptoms=None, confirmed_symptoms=None):
    if denied_symptoms is None:
        denied_symptoms = []
    if confirmed_symptoms is None:
        confirmed_symptoms = []
    predictions, *_ = predict_disease(user_text)
    top_diseases = [pred['disease'].lower() for pred in predictions[:3]]
    candidate_symptoms = set()
    for disease in top_diseases:
        candidate_symptoms.update(DISEASE_SYMPTOMS.get(disease, []))
    mentioned = set(word for word in candidate_symptoms if word in user_text.lower())
    already_asked = set(denied_symptoms) | set(confirmed_symptoms) | mentioned
    remaining = candidate_symptoms - already_asked
    if remaining:
        symptom = list(remaining)[0]
        return f"Are you experiencing '{symptom}'?", symptom
    else:
        return "Can you describe any other symptoms you have?", None
