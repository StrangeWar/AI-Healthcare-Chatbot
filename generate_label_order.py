import pandas as pd
import json

def generate_label_order():
    df = pd.read_csv('realistic_synthetic_symptom_disease_dataset.csv')
    labels = sorted(df['label'].astype(str).unique().tolist())  # enforce sorted order
    with open('label_order.json', 'w') as f:
        json.dump(labels, f)
    print(f"label_order.json generated with {len(labels)} labels in sorted order.")

if __name__ == "__main__":
    generate_label_order()
