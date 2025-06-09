import os
import pandas as pd

def load_imdb_data(base_path):
    data = []
    for label_type in ['pos', 'neg']:
        dir_name = os.path.join(base_path, label_type)
        for file_name in os.listdir(dir_name):
            with open(os.path.join(dir_name, file_name), encoding='utf-8') as f:
                content = f.read()
                label = 1 if label_type == 'pos' else 0
                data.append((content, label))
    return pd.DataFrame(data, columns=['text', 'label'])

# Load both train and test sets
train_df = load_imdb_data('data/raw/aclImdb/train')
test_df = load_imdb_data('data/raw/aclImdb/test')

full_df = pd.concat([train_df, test_df], ignore_index=True)

# Save to CSV
os.makedirs('data/processed', exist_ok=True)
full_df.to_csv('data/processed/cleaned.csv', index=False)
print("[INFO] Saved full dataset to data/processed/cleaned.csv")
