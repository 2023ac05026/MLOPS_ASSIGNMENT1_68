# mlflow.Dockerfile
FROM python:3.11-slim
RUN pip install mlflow "sqlalchemy<2.0" pymysql
EXPOSE 5000
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:///mlruns/mlflow.db", \
     "--default-artifact-root", "./mlruns"]