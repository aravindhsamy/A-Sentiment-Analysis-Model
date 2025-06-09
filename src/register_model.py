import mlflow
from mlflow.tracking import MlflowClient

def register_model(run_id: str, model_name: str):
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"

    # Try to create the registered model if it doesn't exist
    try:
        client.create_registered_model(model_name)
        print(f"Registered new model: {model_name}")
    except Exception as e:
        print(f"Model may already exist: {e}")

    # Create a new model version under the registered model
    model_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id
    )
    print(f"Model version {model_version.version} registered under {model_name}")

if __name__ == "__main__":
    # Replace with your actual run ID and desired model name
    RUN_ID = "1e42e35518a74eceb25e21a9ddf8fb4a"
    MODEL_NAME = "sentiment-analysis-model"

    register_model(RUN_ID, MODEL_NAME)
