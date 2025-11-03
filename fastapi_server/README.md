# TSM MachLeData - Flight Price Prediction

ML pipeline with continuous training, monitoring, and automated deployment.

## Dataset
[Kaggle - Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)

## Features
- CI/CD with GitHub Actions
- Data drift simulation (weekly splits)
- Continuous model retraining
- Performance & drift monitoring
- Model optimization (pruning, quantization)
- REST API deployment

## Setup docker (local environment)

```
# build image
docker build -t flight-api -f docker/Dockerfile .

# run container
docker run -p 8000:8000 flight-api

# access API
http://localhost:8000/docs
```
