import pandas as pd
import random
from transformers import pipeline

def paraphrase(text, n=3):
    try:
        paraphraser = pipeline('text2text-generation', model='eugenesiow/bart-paraphrase')
        results = paraphraser([f"paraphrase: {text}" for _ in range(n)], max_length=60, num_return_sequences=n)
        # Handle both possible output formats
        if isinstance(results[0], dict) and 'generated_text' in results[0]:
            return [r['generated_text'] for r in results]
        else:
            return [str(r) for r in results]
    except Exception as e:
        print(f"Paraphrasing failed: {e}")
        return []

def augment_symptoms(input_csv, output_csv, n_paraphrases=3):
    df = pd.read_csv(input_csv)
    new_rows = []
    for idx, row in df.iterrows():
        text, label = row['text'], row['label']
        new_rows.append({'text': text, 'label': label})
        # Generate paraphrases
        for para in paraphrase(text, n=n_paraphrases):
            new_rows.append({'text': para, 'label': label})
    aug_df = pd.DataFrame(new_rows)
    aug_df.to_csv(output_csv, index=False)
    print(f"Augmented data saved to {output_csv} (original: {len(df)}, augmented: {len(aug_df)})")

if __name__ == "__main__":
    # Example usage: augment symptoms.csv and save to symptoms_augmented.csv
    augment_symptoms('symptoms.csv', 'symptoms_augmented.csv', n_paraphrases=2)
