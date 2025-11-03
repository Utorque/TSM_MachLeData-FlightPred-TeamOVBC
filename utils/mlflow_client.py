import mlflow
import requests

def log_train_data(training_dict, curr_week):
    with mlflow.start_run(run_name=f"week_{curr_week}"):
        mlflow.log_params(training_dict)
        mlflow.log_param("week", curr_week)
    return "training"

def post_new_model(MLFLOW_SERVER, curr_week):
    response = requests.post(
        f"{MLFLOW_SERVER}/models",
        json={"week": curr_week}
    )
    return response
