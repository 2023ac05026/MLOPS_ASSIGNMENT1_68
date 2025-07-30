FROM python:3.11-slim

WORKDIR /pipeline

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src
COPY data/ ./data

# This command will run all your scripts in sequence
CMD ["sh", "-c", "python src/process_data.py && python src/train_logistic_regression.py && python src/train_random_forest.py && python src/promote_best_model.py"]
