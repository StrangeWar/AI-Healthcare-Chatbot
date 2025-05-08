import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load label order for consistent encoding
with open('label_order.json') as f:
    label_order = json.load(f)
label2id = {label: idx for idx, label in enumerate(label_order)}
id2label = {idx: label for idx, label in enumerate(label_order)}

# Load dataset
df = pd.read_csv('realistic_synthetic_symptom_disease_dataset.csv')
df = df[df['label'].isin(label_order)]  # Ensure only known labels
print(df.head())

# Encode labels
df['label_id'] = df['label'].map(label2id)

# Load PubMedBERT
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
    problem_type="single_label_classification"
).to(device)

# Tokenize
def preprocess(examples):
    return tokenizer(examples['symptoms'], truncation=True, padding=True)

dataset = Dataset.from_pandas(df[['symptoms', 'label_id']].rename(columns={'label_id': 'labels'}))
dataset = dataset.map(preprocess, batched=True)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
    fp16=torch.cuda.is_available(),
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train
train_result = trainer.train()
metrics = trainer.evaluate()
print(metrics)

# Save metrics
with open("training_metrics.txt", "w") as f:
    f.write(str(metrics))

# Save final model and tokenizer
model_save_path = "pubmedbert-finetuned"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model and tokenizer saved to {model_save_path}")

# Plotting
history = trainer.state.log_history
train_loss = [x['loss'] for x in history if 'loss' in x and 'epoch' in x]
eval_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]
epochs = sorted(set(x['epoch'] for x in history if 'epoch' in x))

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_loss)+1), train_loss, label='Train Loss')
plt.plot(range(1, len(eval_loss)+1), eval_loss, label='Eval Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss over Epochs')
plt.legend()
plt.savefig('loss_curve.png')
plt.show()

# Confusion matrix
preds = np.argmax(trainer.predict(dataset['test']).predictions, axis=-1)
labels = np.array(dataset['test']['labels'])
cm = confusion_matrix(labels, preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_order, yticklabels=label_order)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()
