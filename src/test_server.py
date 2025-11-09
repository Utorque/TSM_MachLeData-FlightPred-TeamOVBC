import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health Check: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_predict():
    """Test prediction with realistic flight data"""
    payload = {
        "airline": "Air India",
        "ch_code": "AI",
        "num_code": 868,
        "from": "Delhi",
        "to": "Mumbai",
        "Class": "Business",
        "dayofweek": 4,  # Friday (0=Monday)
        "dep_hour": 18,
        "arr_hour": 20,
        "duration_min": 120,  # 2h 00m
        "stops_n": 0
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Prediction: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_predict_economy():
    """Test prediction with economy class"""
    payload = {
        "airline": "Vistara",
        "ch_code": "UK",
        "num_code": 985,
        "from": "Delhi",
        "to": "Mumbai",
        "Class": "Economy",
        "dayofweek": 4,
        "dep_hour": 19,
        "arr_hour": 22,
        "duration_min": 130,
        "stops_n": 0
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Prediction (Economy): {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_predict_with_stop():
    """Test prediction with 1 stop"""
    payload = {
        "airline": "Air India",
        "ch_code": "AI",
        "num_code": 531,
        "from": "Delhi",
        "to": "Mumbai",
        "Class": "Business",
        "dayofweek": 4,
        "dep_hour": 20,
        "arr_hour": 20,
        "duration_min": 1485,  # 24h 45m
        "stops_n": 1
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Prediction (1 stop): {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_current_model():
    """Get current model info"""
    response = requests.get(f"{BASE_URL}/model/current")
    print(f"Current Model: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_list_models():
    """List all registered models"""
    response = requests.get(f"{BASE_URL}/models/list")
    print(f"List Models: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_reload_model():
    """Reload production model"""
    response = requests.post(f"{BASE_URL}/model/reload")
    print(f"Reload Model: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_invalid_predict():
    """Test with invalid data"""
    payload = {
        "airline": "Air India",
        "ch_code": "AI",
        "num_code": 868,
        "from": "Delhi",
        "to": "Mumbai",
        "Class": "Business",
        "dayofweek": 10,  # Invalid: should be 0-6
        "dep_hour": 18,
        "arr_hour": 20,
        "duration_min": 120,
        "stops_n": 0
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Invalid Request: {response.status_code}")
    print(response.text)
    print()

if __name__ == "__main__":
    try:
        print("=" * 60)
        print("Testing Flight Price Prediction API")
        print("=" * 60)
        print()
        
        test_health()
        test_current_model()
        test_list_models()
        
        print("=" * 60)
        print("Prediction Tests")
        print("=" * 60)
        print()
        
        test_predict()
        test_predict_economy()
        test_predict_with_stop()
        
        print("=" * 60)
        print("Management Tests")
        print("=" * 60)
        print()
        
        test_reload_model()
        
        print("=" * 60)
        print("Error Handling")
        print("=" * 60)
        print()
        
        test_invalid_predict()
        
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to server. Is it running on port 8000?")