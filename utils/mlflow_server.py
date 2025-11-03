from .data_loader import load_data
import mlflow

def test_and_promote(curr_week):
    test_data = load_data(curr_week + 1)

    # Evaluate model on test data
    mlflow.log_metric("test_week", curr_week + 1)

    # Promote model if metrics pass threshold
    # This is where comparison between models happens
    return test_data
