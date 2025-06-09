# A-Sentiment-Analysis-Model

**Deploy a Sentiment Analysis Model with a full MLOps pipeline**

---

## Overview

This project demonstrates an end-to-end MLOps workflow for a sentiment analysis model using:

- **Data versioning** with [DVC](https://dvc.org/)
- **Experiment tracking & model registry** with [MLflow](https://mlflow.org/)
- **Model training** on the IMDB dataset
- **Model deployment** as a REST API via MLflow serving
- **Reproducibility, automation, and scalability** for production-ready ML

---

## Features

- Preprocess and clean raw IMDB movie review data
- Train Logistic Regression model with TF-IDF features
- Track parameters, metrics, and models in MLflow UI
- Register and version models in MLflow Model Registry
- Serve models via REST API for real-time prediction
- Manage datasets and models with DVC for version control
- (Optional) Automate pipeline with CI/CD tools like GitHub Actions

---

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- [DVC](https://dvc.org/)
- [MLflow](https://mlflow.org/)

### Installation

```bash
git clone https://github.com/yourusername/A-Sentiment-Analysis-Model.git
cd A-Sentiment-Analysis-Model

pip install -r requirements.txt
dvc pull  # pull dataset and models tracked by DVC
