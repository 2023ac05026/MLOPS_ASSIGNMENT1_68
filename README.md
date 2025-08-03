a# MLOPS_ASSIGNMENT1_68


2023ac05026/rWwBjrFz46gDdy5
https://app.docker.com/accounts/2023ac05026
http://127.0.0.1:5001/view_logs
http://localhost:5001/logs_json 
http://localhost:5001/view_metrics_history
http://localhost:5001/metrics
http://localhost:5001/metrics_ui
http://localhost:5001/predict

MLFLOW : http://localhost:5000




------------------------------------------
docker build -t 2023ac05026/iris-predictor-app ./app

docker push 2023ac05026/iris-predictor-app

docker run -p 5001:5001 -v "%cd%/mlruns:/mlruns" 2023ac05026/iris-predictor-app

docker run -p 5001:5001 2023ac05026/iris-predictor-app


docker build -t 2023ac05026/iris-predictor-app -f app/Dockerfile



docker build -t 2023ac05026/iris-predictor-app ./app

docker run -p 5001:5001 2023ac05026/iris-predictor-app
docker push 2023ac05026/iris-predictor-app


docker-compose up --build -d

docker-compose up -d

docker-compose up -d

docker logs mlflow_server

docker-compose down



# Rebuild the training image
docker build -t 2023ac05026/training-pipeline -f training.Dockerfile .

# Push the updated image
docker push 2023ac05026/training-pipeline

docker run --rm --network iris_default -e MLFLOW_TRACKING_URI=http://mlflow_server:5000 2023ac05026/training-pipeline

docker run --rm \
  --network iris_default \
  -e MLFLOW_TRACKING_URI=http://mlflow_server:5000 \
  -v "$(pwd)/mlruns:/pipeline/mlruns" \
  -v "$(pwd)/app/models:/pipeline/app/models" \
  2023ac05026/training-pipeline


http://127.0.0.1:5001/view_logs
http://localhost:5001/logs_json 
http://localhost:5001/view_metrics_history
http://localhost:5001/metrics
http://localhost:5001/metrics_ui
http://localhost:5001/predict