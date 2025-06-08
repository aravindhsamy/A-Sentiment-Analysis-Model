import os
import pandas as pd

def load_reviews(path, sentiment):
    data = []
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'r', encoding='utf-8') as file:
            review = file.read()
            data.append((review, sentiment))
    return data

base_path = "data/raw/aclImdb"
train_pos = load_reviews(os.path.join(base_path, "train/pos"), "positive")
train_neg = load_reviews(os.path.join(base_path, "train/neg"), "negative")
test_pos  = load_reviews(os.path.join(base_path, "test/pos"), "positive")
test_neg  = load_reviews(os.path.join(base_path, "test/neg"), "negative")

# Combine all
all_reviews = train_pos + train_neg + test_pos + test_neg
df = pd.DataFrame(all_reviews, columns=["review", "sentiment"])
df = df.sample(frac=1, random_state=42)  # Shuffle

# Save to CSV
output_path = "data/raw/imdb.csv"
df.to_csv(output_path, index=False)
print(f"Saved combined dataset to: {output_path}")
