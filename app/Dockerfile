# app/Dockerfile
# Use a lightweight official Python image as a base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file from the build context (which is the 'app' directory)
COPY requirements.txt .

# Install the Python dependencies and Gunicorn
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy the rest of the application source code into the container
# This is now more explicit to ensure the app folder is copied
COPY . .

# Expose the port the app will run on
EXPOSE 5001

# Define the command to run the application using Gunicorn
# The command assumes your main Flask app instance is named 'app'
# It starts Gunicorn, binds it to all network interfaces on port 5001,
# and specifies the application to run as 'Iris_predictor:app'
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "Iris_predictor:app"]