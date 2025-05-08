import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import json

# Load label order
with open('label_order.json') as f:
    label_order = json.load(f)
label2id = {label: idx for idx, label in enumerate(label_order)}
id2label = {idx: label for label, idx in label2id.items()}

# Load model and tokenizer
MODEL_PATH = './pubmedbert-finetuned'
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Load dataset
df = pd.read_csv('realistic_synthetic_symptom_disease_dataset.csv')
df = df[df['label'].isin(label_order)]  # Ensure only known labels

# Encode labels
df['label_id'] = df['label'].map(label2id)

# Split into train/test
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_id'])

# Predict on test set
preds = []
true = []
for _, row in test_df.iterrows():
    inputs = tokenizer(row['symptoms'], return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_id = int(torch.argmax(logits, dim=1).cpu().numpy()[0])
    preds.append(pred_id)
    true.append(row['label_id'])

# Metrics
print("Accuracy:", accuracy_score(true, preds))
print("\nClassification Report:\n", classification_report(true, preds, target_names=label_order))
print("\nConfusion Matrix:\n", confusion_matrix(true, preds))