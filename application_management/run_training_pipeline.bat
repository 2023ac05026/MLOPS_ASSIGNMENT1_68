@echo off
SETLOCAL

REM --- CONFIGURATION ---
REM IMPORTANT: Replace "iris_default" with your project's actual network name.
REM Find it by running "docker network ls" after starting the services.
SET NETWORK_NAME=iris_default

REM Replace this with your Docker Hub username
SET DOCKER_USERNAME=2023ac05026

REM --- EXECUTION ---
ECHO.
ECHO Starting the training pipeline...
ECHO This will connect to the running MLflow server and log the results.
ECHO.

REM Pull the latest version of the training image from Docker Hub
docker pull %DOCKER_USERNAME%/training-pipeline:latest

REM Run the training container
docker run --rm ^
  --network %NETWORK_NAME% ^
  -e MLFLOW_TRACKING_URI=http://mlflow_server:5000 ^
  -v "%cd%\mlruns:/pipeline/mlruns" ^
  -v "%cd%\app\models:/pipeline/app/models" ^
  %DOCKER_USERNAME%/training-pipeline

ECHO.
ECHO --- Training Pipeline Finished ---
ECHO You can now view the new experiments in the MLflow UI.
ECHO The new model.joblib file has been saved to the app/models directory.
ECHO.
PAUSE