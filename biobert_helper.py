from typing import List, Dict, Set, Tuple, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, pipeline
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

class ConversationState:
    def __init__(self):
        self.confirmed_symptoms: Set[str] = set()
        self.denied_symptoms: Set[str] = set()
        self.conversation_history: List[Dict] = []
        self.current_confidence: float = 0.0
        self.top_predictions: List[Dict] = []

    def add_interaction(self, symptoms_text: str, predictions: List[Dict]):
        self.conversation_history.append({
            "input": symptoms_text,
            "predictions": predictions
        })
        if predictions:
            self.top_predictions = predictions[:3]
            self.current_confidence = predictions[0]["confidence"]

    def add_symptom_response(self, symptom: str, is_confirmed: bool):
        if is_confirmed:
            self.confirmed_symptoms.add(symptom)
        else:
            self.denied_symptoms.add(symptom)

conversations: Dict[str, ConversationState] = {}

def get_or_create_conversation(session_id: str) -> ConversationState:
    if session_id not in conversations:
        conversations[session_id] = ConversationState()
    return conversations[session_id]

def predict_disease(symptoms_text: str, session_id: str = None) -> Tuple[List[Dict], str, List[str], str, str]:
    # Get conversation state if session_id provided
    conv_state = get_or_create_conversation(session_id) if session_id else None
    
    # Preprocess input
    clean_text = preprocess_text(symptoms_text)
    extracted = extract_symptoms(symptoms_text)
    intent = detect_intent(symptoms_text)
    sentiment = detect_sentiment(symptoms_text)

    # If we have conversation state, include confirmed symptoms
    if conv_state and conv_state.confirmed_symptoms:
        clean_text += ". " + ", ".join(conv_state.confirmed_symptoms)

    model.eval()
    inputs = tokenizer(clean_text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).squeeze().tolist()
    
    predictions = [
        {'disease': label, 'confidence': float(round(prob * 100, 1)) if prob <= 1.0 else float(round(prob, 1))}
        for label, prob in zip(DISEASE_LABELS, probs)
    ]
    predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    # Update conversation state
    if conv_state:
        conv_state.add_interaction(symptoms_text, predictions)
    
    top = predictions[0]
    threshold = 20.0  # 20% confidence threshold
    
    # Adjust explanation based on conversation state
    if conv_state and conv_state.conversation_history:
        if top['confidence'] < threshold:
            explanation = (
                f"Based on our conversation so far, I'm not confident enough about the diagnosis. "
                f"I'll ask you some more questions to better understand your symptoms."
            )
        else:
            previous_confidence = conv_state.conversation_history[-2]['predictions'][0]['confidence'] if len(conv_state.conversation_history) > 1 else 0
            confidence_change = top['confidence'] - previous_confidence
            
            if confidence_change > 5:
                explanation = (
                    f"My confidence has increased! Based on all symptoms described, "
                    f"the most likely condition is {top['disease']} "
                    f"(confidence: {top['confidence']}%). Here are other possibilities:\n"
                    + ' • '.join([f"{p['disease']} ({p['confidence']}%)" for p in predictions[1:3]])
                )
            else:
                explanation = (
                    f"Based on all symptoms described, the most likely condition is {top['disease']} "
                    f"(confidence: {top['confidence']}%). Other possibilities include:\n"
                    + ' • '.join([f"{p['disease']} ({p['confidence']}%)" for p in predictions[1:3]])
                )
    else:
        if top['confidence'] < threshold:
            explanation = "I need more information about your symptoms to make a confident prediction. I'll ask you some questions to help better understand your condition."
        else:
            explanation = (
                f"Based on your initial description, the most likely condition could be {top['disease']} "
                f"(confidence: {top['confidence']}%). Let me ask you a few questions to confirm."
            )

    return predictions, explanation, extracted, intent, sentiment

def get_followup_question(user_text: str, session_id: str, denied_symptoms: List[str] = None, confirmed_symptoms: List[str] = None) -> Tuple[str, Optional[str]]:
    conv_state = get_or_create_conversation(session_id)
    
    # Update conversation state with provided symptoms
    if denied_symptoms:
        conv_state.denied_symptoms.update(denied_symptoms)
    if confirmed_symptoms:
        conv_state.confirmed_symptoms.update(confirmed_symptoms)
    
    # Get predictions if we don't have any
    if not conv_state.top_predictions:
        predictions, *_ = predict_disease(user_text, session_id)
        top_diseases = [pred['disease'] for pred in predictions[:3]]
    else:
        top_diseases = [pred['disease'] for pred in conv_state.top_predictions]

    # Gather candidate symptoms from top diseases
    candidate_symptoms = set()
    for disease in top_diseases:
        candidate_symptoms.update(DISEASE_SYMPTOMS.get(disease, []))
    
    # Remove already addressed symptoms
    mentioned = set(word.lower() for word in candidate_symptoms if word.lower() in user_text.lower())
    already_asked = conv_state.denied_symptoms | conv_state.confirmed_symptoms | mentioned
    remaining = candidate_symptoms - already_asked    # If confidence is very high and we've confirmed key symptoms, we can stop
    if (conv_state.current_confidence >= 85.0 and 
        len(conv_state.confirmed_symptoms) >= 3 and 
        len(conv_state.conversation_history) > 2):
        return "I'm confident in my assessment. Would you like me to explain my diagnosis in more detail?", None
        
    if remaining:
        # Sort remaining symptoms by relevance to top prediction
        top_disease_symptoms = set(DISEASE_SYMPTOMS.get(top_diseases[0], []))
        priority_symptoms = remaining & top_disease_symptoms
        
        if priority_symptoms:
            symptom = list(priority_symptoms)[0]
        else:
            symptom = list(remaining)[0]
            
        return f"Are you experiencing '{symptom}'?", symptom
    else:
        return "Can you describe any other symptoms you're experiencing?", None
