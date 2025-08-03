@echo off
ECHO --- Pulling the latest prediction app image from Docker Hub ---
docker-compose pull iris_predictor_app

ECHO.
ECHO --- Starting MLflow server and prediction app ---
ECHO This will build the MLflow server image if it doesn't exist.

REM Start the services defined in docker-compose.yml in detached mode (-d)
REM --build ensures that any changes to the mlflow.Dockerfile are applied
docker-compose up --build -d

ECHO.
ECHO --- Services started ---
ECHO MLflow UI should be accessible at: http://localhost:5000
ECHO Prediction App should be accessible at: http://localhost:5001
ECHO.
PAUSE