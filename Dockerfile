FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for layer caching
COPY fastapi_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project structure to maintain imports
COPY . /app/

# Set Python path
ENV PYTHONPATH=/app
ENV DATA_PATH=/app/data/Flights.csv
ENV MLFLOW_TRACKING_URI=sqlite:///mlruns.db

# Expose port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "fastapi_server.src.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]