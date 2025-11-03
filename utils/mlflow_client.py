import mlflow
import mlflow.sklearn
import requests

def log_train_data(training_dict, curr_week):
    with mlflow.start_run(run_name=f"week_{curr_week}") as run:
        for key, value in training_dict.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
            else:
                mlflow.log_param(key, value)
        mlflow.log_param("training_week", curr_week)
        mlflow.log_metric("num_samples", training_dict.get("num_samples", 0))
    return "training"

def post_new_model(MLFLOW_SERVER, curr_week):
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    model_name = f"flight_model_week_{curr_week}"

    result = mlflow.register_model(model_uri, model_name)

    response = requests.post(
        f"{MLFLOW_SERVER}/models/register",
        json={
            "week": curr_week,
            "model_name": model_name,
            "model_version": result.version
        }
    )
    return response
