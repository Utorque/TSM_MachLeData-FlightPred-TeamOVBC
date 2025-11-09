import mlflow
import mlflow.sklearn
import requests
import joblib
import io
import base64

def log_model_train_info(training_dict, curr_week):
    with mlflow.start_run(run_name=f"log_train_week_{curr_week}", nested=True):
        for key, value in training_dict.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
            else:
                mlflow.log_param(key, value)
        mlflow.log_param("training_week", curr_week)
        mlflow.log_metric("num_samples", training_dict.get("num_samples", 0))
    return "training"

def post_new_model(server_url, curr_week, model):
    """
    Serialize model and send to FastAPI server for registration and promotion
    """
    # Serialize model to bytes
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    model_bytes = buffer.read()

    # Encode to base64 for JSON transport
    model_b64 = base64.b64encode(model_bytes).decode('utf-8')

    # Send to server
    response = requests.post(
        f"{server_url}/model/upload",
        json={
            "week": curr_week,
            "model_data": model_b64
        },
        timeout=60
    )

    response.raise_for_status()
    return response.json()
