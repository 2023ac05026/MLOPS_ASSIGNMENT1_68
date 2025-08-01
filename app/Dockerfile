# app/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001

# This command explicitly tells Flask to run the 'app' object
# from the 'Iris_predictor.py' file.
CMD ["flask", "run", "--app", "Iris_predictor.py", "--host", "0.0.0.0", "--port", "5001"]