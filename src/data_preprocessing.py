import pandas as pd
import re
import os

INPUT_PATH = 'data/raw/sentiment.csv'
OUTPUT_PATH = 'data/processed/cleaned.csv'

def clean_text(text):
    text = re.sub(r"http\S+|[^a-zA-Z\s]", '', text)
    return text.lower().strip()

def preprocess():
    df = pd.read_csv(INPUT_PATH)
    
    # Assuming your dataset has 'text' and 'label' columns
    df['text'] = df['text'].astype(str).apply(clean_text)
    df = df.dropna()
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[INFO] Cleaned data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess()
