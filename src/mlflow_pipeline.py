from utils.mlflow_client import log_model_train_info, post_new_model
from utils.mlflow_drift import check_and_log_drift
from utils.data_loader import load_data
from models.baseModel import train_model
import mlflow

DATA_DRIFT = False
CONCEPT_DRIFT = True
CURR_WEEK = 8

mlflow.set_experiment('Flight_Model_Training')

with mlflow.start_run(run_name=f'week_{CURR_WEEK}') as parent_run:

    # Load data
    train_data, training_dict = load_data(upto_week=CURR_WEEK, data_drift=DATA_DRIFT, concept_drift=CONCEPT_DRIFT)

    # Check drift
    retrain_trigger = check_and_log_drift(train_data, current_week=CURR_WEEK)

    # Log training config
    log_model_train_info(training_dict, curr_week=CURR_WEEK)

    # Train if necessary
    if retrain_trigger:
        print('Drift detected retrain model')
        # TODO adapt following part
        model = train_model(train_data,CURR_WEEK)
        # evaluate_model(model)
        # post_new_model('http://mlflow-server:5000', curr_week=CURR_WEEK)
    else:
        print('No significant drift - do not retrain model')
