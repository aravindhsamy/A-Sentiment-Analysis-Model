import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Load data
df = pd.read_csv("data/raw/imdb.csv")
X = df["review"]
y = df["sentiment"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline (vectorizer + classifier)
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ("clf", LogisticRegression(solver="liblinear", max_iter=1000))
])

# Start MLflow run
with mlflow.start_run():

    # Train pipeline
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Log model with preprocessing
    signature = infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(pipeline, "model", signature=signature)

    # Log parameters and metric
    mlflow.log_params({
        "model_type": "LogisticRegression",
        "vectorizer": "TfidfVectorizer",
        "max_features": 5000,
        "ngram_range": "(1,2)",
        "solver": "liblinear"
    })
    mlflow.log_metric("accuracy", acc)

    print(f"✅ Model trained and logged with accuracy: {acc:.4f}")
