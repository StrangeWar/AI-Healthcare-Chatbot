# AI Healthcare Chatbot

## Overview
This project is an AI-powered healthcare chatbot that predicts possible diseases based on user-described symptoms. It leverages a fine-tuned PubMedBERT model for disease classification and provides an interactive web interface for users to chat with the bot, receive predictions, and get follow-up questions for more accurate results.

## Features
- **Conversational Symptom Checker:** Users can describe symptoms in natural language and receive possible disease predictions.
- **Follow-up Questions:** The chatbot asks clarifying questions if the model's confidence is low.
- **Frontend (React + Vite):** Modern, responsive chat UI.
- **Backend (Flask):** Handles prediction requests and model inference.
- **Fine-tuned PubMedBERT Model:** Trained on a synthetic symptom-disease dataset for robust predictions.
- **Customizable Disease/Symptom Mapping:** Easily extendable for new diseases or symptoms.
- **Advanced NLP Features:**
  - Symptom extraction
  - Intent detection
  - Sentiment analysis
  - Spell correction
  - Symptom synonym mapping

## Architecture
- **Frontend:**
  - Built with React and Vite (`src/`)
  - Communicates with backend via REST API
  - Handles chat logic, displays predictions, and manages follow-up questions
- **Backend:**
  - Python Flask app (`app.py`)
  - Loads fine-tuned PubMedBERT model (`pubmedbert-finetuned/`)
  - Uses helper functions (`biobert_helper.py`) for preprocessing and prediction
  - Exposes endpoints for prediction and follow-up
- **Model:**
  - Fine-tuned PubMedBERT for sequence classification
  - Model and tokenizer saved in `pubmedbert-finetuned/`
- **Dataset:**
  - `realistic_synthetic_symptom_disease_dataset.csv` contains synthetic symptom-disease pairs
  - Data augmentation supported via `augment_symptoms.py`
  - `label_order.json` defines disease label order for consistent encoding

## Setup & Installation
1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd <repo-folder>
   ```
2. **Backend Setup (Python):**
   - Install dependencies:
     ```sh
     pip install -r requirements.txt
     ```
   - Start the Flask backend:
     ```sh
     python app.py
     ```
3. **Frontend Setup (React):**
   - Install dependencies:
     ```sh
     npm install
     ```
   - Start the frontend:
     ```sh
     npm run dev
     ```
   - The app will be available at `http://localhost:5173` (default Vite port).

## Usage
- Open the frontend in your browser.
- Type your symptoms in the chat and send.
- The bot will reply with possible conditions and may ask follow-up questions.
- If the model is highly confident (>90%), it will provide a direct prediction.
- If not, it will ask for more details to improve accuracy.

## File Structure
- `app.py` — Flask backend server
- `biobert_helper.py` — Model loading, preprocessing, and prediction logic
- `finetune_biobert.py` — Script to fine-tune PubMedBERT on the dataset
- `evaluate_pubmedbert.py` — Model evaluation script
- `analyze_dataset.py` — Dataset analysis and statistics
- `augment_symptoms.py` — Data augmentation using paraphrasing
- `generate_label_order.py` — Generate consistent disease label ordering
- `realistic_synthetic_symptom_disease_dataset.csv` — Training/evaluation data
- `label_order.json` — Disease label mapping
- `pubmedbert-finetuned/` — Fine-tuned model and tokenizer files
- `src/` — React frontend source code

## Model & Dataset Details
- **Model:** 
  - Base: PubMedBERT (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)
  - Fine-tuned for multi-class disease classification
  - Includes confidence thresholds for reliable predictions
- **Training:** 
  - See `finetune_biobert.py` for training pipeline and metrics
  - Supports data augmentation via paraphrasing
  - Uses weighted cross-entropy loss for balanced training
- **Evaluation:** 
  - See `evaluate_pubmedbert.py` for accuracy, classification report, and confusion matrix
  - Dataset analysis tools in `analyze_dataset.py`
- **Dataset:** 
  - Synthetic, realistic symptom-disease pairs
  - Supports automated augmentation
  - Includes common variations and synonyms of symptoms

## Supported Diseases & Symptoms
The chatbot includes mappings for various common conditions including:
- Respiratory: Bronchitis, Asthma, Pneumonia
- Infectious: COVID-19, Flu, Common Cold, Chickenpox
- Neurological: Migraine, Meningitis
- Digestive: Gastroenteritis
- Other: Allergic Rhinitis, Sinusitis, Malaria, Tuberculosis

Each disease has associated key symptoms that help in accurate diagnosis.

## Customization & Extension
- Add new diseases or symptoms by updating the dataset and retraining the model
- Extend frontend UI in `src/` for more features (e.g., hospital recommendations, maps)
- Backend logic can be expanded for more advanced NLP or integration with medical APIs
- Customize symptom synonyms and disease mappings in `biobert_helper.py`

## License
This project is for educational and research purposes. Please consult a healthcare professional for real medical advice.

---

*Generated on June 3, 2025.*
