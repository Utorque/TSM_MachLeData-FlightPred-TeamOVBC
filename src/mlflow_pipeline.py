from utils.mlflow_client import log_model_train_info, post_new_model
from utils.mlflow_drift import check_and_log_drift
from utils.mlflow_model_drift import check_and_log_model_drift
from utils.data_loader import load_data
from models.baseModel import train_model
import mlflow

DATA_DRIFT = False
CONCEPT_DRIFT = False
CURR_WEEK = 9

mlflow.set_experiment('Flight_Model_Training')

with mlflow.start_run(run_name=f'week_{CURR_WEEK}') as parent_run:

    # Load data
    train_data, training_dict = load_data(upto_week=CURR_WEEK, data_drift=DATA_DRIFT, concept_drift=CONCEPT_DRIFT, path='data/Flights.csv')

    # Check drift
    retrain_trigger = check_and_log_drift(train_data, current_week=CURR_WEEK)

    # Log training config
    log_model_train_info(training_dict, curr_week=CURR_WEEK)

    # Train if necessary
    if retrain_trigger:
        print('Drift detected, retrain model')
        
        # Unpack the tuple
        model, metrics, train_weeks, test_week = train_model(train_data, CURR_WEEK)
        
        # Log metrics to MLflow
        with mlflow.start_run(run_name=f"metrics_week_{CURR_WEEK}", nested=True):
            mlflow.log_metrics(metrics)
            mlflow.log_param("train_weeks", str(train_weeks))
            mlflow.log_param("test_week", test_week)
        
        # Post ONLY the model (not the tuple)
        result = post_new_model(
            server_url="http://localhost:52001",
            curr_week=CURR_WEEK,
            model=model  # Pass only the model, not the tuple
        )
        
        print(f"Upload result: {result}")
    else:
        print('No significant drift - do not retrain model')

        # Log model drift here
        print('Check model drift')
        check_and_log_model_drift(train_data, current_week=CURR_WEEK)
