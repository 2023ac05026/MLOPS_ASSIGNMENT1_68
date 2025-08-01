# app/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements from the build context (which is now './app')
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything from the build context (the 'app' folder)
# This includes Iris_predictor.py, templates/, and the new models/ folder
COPY . .

EXPOSE 5001

ENV FLASK_APP=Iris_predictor.py

CMD ["flask", "run", "--host", "0.0.0.0", "--port", "5001"]