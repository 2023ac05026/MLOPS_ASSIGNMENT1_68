import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend


# Enable MLflow autologging
mlflow.autolog()

# Start run
with mlflow.start_run():

    # Read dataset
    df = pd.read_csv("Data/processed/processed_iris.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Manual logging (optional)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(model, "random_forest_model")

    # Optionally log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    print("Model saved to run:", mlflow.active_run().info.run_id)
