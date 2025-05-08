import pandas as pd
from collections import Counter

# Load dataset
csv_path = 'realistic_synthetic_symptom_disease_dataset.csv'
df = pd.read_csv(csv_path)

# Label distribution
label_counts = df['label'].value_counts()
print('Label distribution:')
print(label_counts)
print('\n')

# Most common symptom patterns for each label
def most_common_symptoms_per_label(df, top_n=5):
    for label in df['label'].unique():
        subset = df[df['label'] == label]
        print(f'--- {label} ({len(subset)}) ---')
        # Show most common symptom patterns
        print(subset['symptoms'].value_counts().head(top_n))
        print()

most_common_symptoms_per_label(df)
