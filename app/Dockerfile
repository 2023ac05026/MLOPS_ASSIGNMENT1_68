# app/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001

# Set the FLASK_APP environment variable to point to your main file.
# This is the most reliable way to specify the application.
ENV FLASK_APP=Iris_predictor.py

# The command now simply runs flask, which will automatically pick up
# the FLASK_APP variable.
CMD ["flask", "run", "--host", "0.0.0.0", "--port", "5001"]