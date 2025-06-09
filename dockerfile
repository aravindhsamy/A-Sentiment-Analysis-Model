# Use official Python image as base
FROM python:3.8-slim

# Install MLflow and dependencies
RUN pip install mlflow scikit-learn pandas

# Copy your model files or pull model from MLflow Model Registry
# For simplicity, copy the whole project (adjust as needed)
COPY . /app
WORKDIR /app

# Expose the port MLflow server will use
EXPOSE 1234

# Command to serve your registered model (adjust model name and version)
CMD ["mlflow", "models", "serve", "-m", "models:/sentiment-analysis-model/1", "-p", "1234", "--no-conda"]
