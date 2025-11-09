from data_loader import load_data
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, r2_score

def test_and_promote(curr_week):
    test_data = load_data(curr_week + 1)

    model_name = f"flight_model_week_{curr_week}"
    baseline_model_name = f"flight_model_week_6"

    # Load current model
    current_model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")

    # Load baseline model (week 6)
    if curr_week > 6:
        baseline_model = mlflow.sklearn.load_model(f"models:/{baseline_model_name}/latest")
    else:
        baseline_model = None

    # Prepare test data
    X_test = test_data.drop(columns=['price'])
    y_test = test_data['price']

    # Evaluate current model
    y_pred_current = current_model.predict(X_test)
    mse_current = mean_squared_error(y_test, y_pred_current)
    r2_current = r2_score(y_test, y_pred_current)

    with mlflow.start_run(run_name=f"test_week_{curr_week + 1}"):
        mlflow.log_metric("test_week", curr_week + 1)
        mlflow.log_metric("mse_current", mse_current)
        mlflow.log_metric("r2_current", r2_current)

        # Compare with baseline if available
        if baseline_model is not None:
            y_pred_baseline = baseline_model.predict(X_test)
            mse_baseline = mean_squared_error(y_test, y_pred_baseline)
            r2_baseline = r2_score(y_test, y_pred_baseline)

            mlflow.log_metric("mse_baseline", mse_baseline)
            mlflow.log_metric("r2_baseline", r2_baseline)
            mlflow.log_metric("drift_score", abs(mse_current - mse_baseline))

        # Promote if metrics are acceptable
        if r2_current > 0.7:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version="latest",
                stage="Production"
            )

    return test_data
