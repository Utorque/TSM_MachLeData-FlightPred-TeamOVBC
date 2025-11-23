from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from typing import Optional, Dict
import os
import sys
from contextlib import asynccontextmanager
import joblib
import io
import base64

DATA_PATH = os.getenv("DATA_PATH", "/app/data/Flights.csv")

# At the top of the file, proper imports
import sys
sys.path.insert(0, '/app')
from utils.mlflow_server import test_and_promote

# Global model cache
model_cache = {"model": None, "model_name": None, "version": None}

def load_production_model():
    """Load the latest production model from MLflow registry"""
    client = MlflowClient()

    try:
        registered_models = client.search_registered_models()

        production_model = None
        latest_week = 0

        for rm in registered_models:
            model_name = rm.name
            if "flight_model_week_" in model_name:
                try:
                    week_num = int(model_name.split("_")[-1])

                    versions = client.search_model_versions(f"name='{model_name}'")
                    for version in versions:
                        if version.current_stage == "Production":
                            if week_num > latest_week:
                                latest_week = week_num
                                production_model = {
                                    "name": model_name,
                                    "version": str(version.version),  # Convert to string
                                    "week": week_num
                                }
                except (ValueError, IndexError):
                    continue

        if production_model:
            model_uri = f"models:/{production_model['name']}/Production"
            model = mlflow.sklearn.load_model(model_uri)

            model_cache["model"] = model
            model_cache["model_name"] = production_model["name"]
            model_cache["version"] = production_model["version"]
            model_cache["week"] = production_model["week"]

            return model
        else:
            # Fallback: load latest model regardless of stage
            for rm in registered_models:
                if "flight_model_week_" in rm.name:
                    try:
                        week_num = int(rm.name.split("_")[-1])
                        if week_num > latest_week:
                            latest_week = week_num
                            model_name = rm.name
                    except (ValueError, IndexError):
                        continue

            if latest_week > 0:
                model_uri = f"models:/{model_name}/latest"
                model = mlflow.sklearn.load_model(model_uri)

                model_cache["model"] = model
                model_cache["model_name"] = model_name
                model_cache["version"] = "latest"
                model_cache["week"] = latest_week

                return model

            raise ValueError("No flight models found in MLflow registry")

    except Exception as e:
        raise RuntimeError(f"Failed to load model from MLflow: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model
    try:
        load_production_model()
        print(f"Loaded model: {model_cache['model_name']} version {model_cache['version']}")
    except Exception as e:
        print(f"Warning: Could not load model on startup: {e}")

    yield

    # Shutdown
    model_cache["model"] = None

app = FastAPI(
    title="Flight Price Prediction API - MLflow Backend",
    description="API with MLflow model registry backend for flight price prediction with drift simulation",
    version="2.0.0",
    lifespan=lifespan
)

class FlightData(BaseModel):
    """Flight data for prediction"""
    airline: str = Field(..., description="Airline name")
    ch_code: str = Field(..., description="Flight code (ch)")
    num_code: int = Field(..., description="Flight number code")
    from_location: str = Field(..., alias="from", description="Departure location")
    to_location: str = Field(..., alias="to", description="Arrival location")
    Class: str = Field(..., description="Flight class (Business/Economy)")
    dayofweek: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    dep_hour: int = Field(..., ge=0, le=23, description="Departure hour")
    arr_hour: int = Field(..., ge=0, le=23, description="Arrival hour")
    time_taken_minutes: float = Field(..., gt=0, description="Flight duration in minutes")
    stops_n: int = Field(..., ge=0, description="Number of stops")

    class Config:
        populate_by_name = True

class PredictionResponse(BaseModel):
    """Prediction response"""
    predicted_price: float = Field(..., description="Predicted price in CHF")
    model_name: str = Field(..., description="Model used for prediction")
    model_version: str = Field(..., description="Model version")
    model_week: int = Field(..., description="Training week")

class ModelInfo(BaseModel):
    """Current model information"""
    model_name: Optional[str] = None
    version: Optional[str] = None
    week: Optional[int] = None
    status: str

class PromoteResponse(BaseModel):
    """Model promotion response"""
    success: bool
    message: str
    promoted_model: Optional[str] = None

class ModelUpload(BaseModel):
    """Model upload request"""
    week: int = Field(..., description="Training week number")
    model_data: str = Field(..., description="Base64 encoded model data")

class ModelUploadResponse(BaseModel):
    """Model upload response"""
    success: bool
    message: str
    model_name: str
    version: Optional[str] = None
    promoted: bool

@app.get("/health")
def health_check():
    """Health check endpoint"""
    model_loaded = model_cache["model"] is not None
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_price(data: FlightData):
    """
    Predict flight price using the current production model
    """
    if model_cache["model"] is None:
        try:
            load_production_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not available: {str(e)}")

    try:
        # Prepare dataframe with exact column names and types expected by model
        input_dict = {
            "airline": [data.airline],
            "ch_code": [data.ch_code],
            "num_code": [data.num_code],
            "from": [data.from_location],
            "to": [data.to_location],
            "Class": [data.Class],
            "dayofweek": [data.dayofweek],
            "dep_hour": [data.dep_hour],
            "arr_hour": [data.arr_hour],
            "time_taken_minutes": [data.time_taken_minutes],
            "stops_n": [data.stops_n]
        }

        df = pd.DataFrame(input_dict)
        
        print("\n=== DEBUG INFO ===")
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame dtypes:\n{df.dtypes}")
        print(f"DataFrame content:\n{df}")
        print(f"Model type: {type(model_cache['model'])}")
        print("==================\n")

        # Predict
        prediction = model_cache["model"].predict(df)[0]

        return PredictionResponse(
            predicted_price=float(prediction),
            model_name=model_cache["model_name"],
            model_version=str(model_cache["version"]),
            model_week=model_cache["week"]
        )

    except Exception as e:
        import traceback
        print("\n=== FULL TRACEBACK ===")
        traceback.print_exc()
        print("======================\n")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model/current", response_model=ModelInfo)
def get_current_model():
    """Get information about the current production model"""
    if model_cache["model"] is None:
        return ModelInfo(status="no_model_loaded")

    return ModelInfo(
        model_name=model_cache["model_name"],
        version=model_cache["version"],
        week=model_cache["week"],
        status="loaded"
    )

@app.post("/model/reload")
def reload_model():
    """Reload the production model from MLflow registry"""
    try:
        load_production_model()
        return {
            "success": True,
            "message": "Model reloaded successfully",
            "model_name": model_cache["model_name"],
            "version": model_cache["version"],
            "week": model_cache["week"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")

@app.post("/model/promote/{week}", response_model=PromoteResponse)
def promote_model(week: int):
    """
    Test and promote a model for a given week using the mlflow_server utils.
    This runs the promotion algorithm and updates the model registry.
    """
    try:
        # Run test and promotion
        test_and_promote(week, DATA_PATH)

        # Reload the model after promotion
        load_production_model()

        return PromoteResponse(
            success=True,
            message=f"Model week {week} tested and promoted if criteria met",
            promoted_model=f"flight_model_week_{week}"
        )

    except Exception as e:
        return PromoteResponse(
            success=False,
            message=f"Promotion failed: {str(e)}, detail: {str(e.with_traceback())}",
            promoted_model=None
        )

@app.post("/model/upload", response_model=ModelUploadResponse)
def upload_model(data: ModelUpload):
    """
    Receive a trained model, register it to MLflow, and run promotion logic.
    If this is the first model or no production model exists, it's automatically promoted.
    """
    try:
        # Deserialize model from base64
        model_bytes = base64.b64decode(data.model_data)
        buffer = io.BytesIO(model_bytes)
        model = joblib.load(buffer)

        model_name = f"flight_model_week_{data.week}"

        # Register model to MLflow
        with mlflow.start_run(run_name=f"register_week_{data.week}"):
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                registered_model_name=model_name
            )
            mlflow.log_param("week", data.week)

        # Get the latest version
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        latest_version = max([int(v.version) for v in versions])

        # Check if there are any production models
        registered_models = client.search_registered_models()
        has_production_model = False

        for rm in registered_models:
            if "flight_model_week_" in rm.name:
                model_versions = client.search_model_versions(f"name='{rm.name}'")
                for v in model_versions:
                    if v.current_stage == "Production":
                        has_production_model = True
                        break
                if has_production_model:
                    break

        promoted = False

        # If no production model exists, promote this one automatically
        if not has_production_model:
            client.transition_model_version_stage(
                name=model_name,
                version=str(latest_version),
                stage="Production"
            )
            promoted = True
            message = f"Model registered and promoted to Production (first model)"
        else:
            # Run promotion logic
            try:
                test_and_promote(data.week, DATA_PATH)
                # Check if it was promoted
                updated_versions = client.search_model_versions(f"name='{model_name}'")
                for v in updated_versions:
                    if v.version == str(latest_version) and v.current_stage == "Production":
                        promoted = True
                        break

                if promoted:
                    message = f"Model registered and promoted to Production (passed criteria)"
                else:
                    message = f"Model registered but not promoted (criteria not met)"

            except Exception as e:
                import traceback
                message = f"Model registered, promotion test failed: {traceback.format_exc()}"


        # Reload production model in cache
        try:
            load_production_model()
        except:
            pass

        return ModelUploadResponse(
            success=True,
            message=message,
            model_name=model_name,
            version=str(latest_version),
            promoted=promoted
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model upload failed: {str(e)}")

@app.get("/models/list")
def list_models():
    """List all registered flight models and their stages"""
    try:
        client = MlflowClient()
        registered_models = client.search_registered_models()

        models_info = []
        for rm in registered_models:
            if "flight_model_week_" in rm.name:
                versions = client.search_model_versions(f"name='{rm.name}'")
                version_info = []
                for v in versions:
                    version_info.append({
                        "version": v.version,
                        "stage": v.current_stage,
                        "creation_time": v.creation_timestamp
                    })

                models_info.append({
                    "name": rm.name,
                    "versions": version_info
                })

        return {"models": models_info}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
