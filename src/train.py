import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('data/processed/cleaned.csv')
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(solver='liblinear'))
])

with mlflow.start_run():
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"[INFO] Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Log params and metrics
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("vectorizer", "TF-IDF")
    mlflow.log_metric("accuracy", acc)

    # Log the model artifact
    mlflow.sklearn.log_model(pipeline, "model")

    # Also save locally as before
    joblib.dump(pipeline, 'models/best_model.pkl')
    print("[INFO] Model saved to models/best_model.pkl")
