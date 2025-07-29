
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Save CSV
df.to_csv(".\\Data\\raw\\iris.csv", index=False)

print("Iris dataset saved to data/iris.csv")


X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')

#Map numerical labels to species names
target_names = {i: name for i, name in enumerate(iris.target_names)}
y_named = y.map(target_names)

target_names = {i: name for i, name in enumerate(iris.target_names)}
y_named = y.map(target_names)

# Step 2: Combine features and target into one DataFrame
df = X.copy()
df['target'] = y_named

# Step 3: Check for missing values
if df.isnull().values.any():
    print("Missing values found. Filling with column mean...")
    df.fillna(df.mean(), inplace=True)
else:
    print("No missing values found.")

# Step 4: Encode categorical target (if needed)
label_encoder = LabelEncoder()
df['target_encoded'] = label_encoder.fit_transform(df['target'])

# Step 5: Feature Scaling (Standardization)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df[iris.feature_names])
df_scaled = pd.DataFrame(features_scaled, columns=iris.feature_names)
df_scaled['target'] = df['target_encoded']

df_scaled.to_csv(".\\Data\\processed\\processed_iris.csv",index=False)

print("processed data saved to \\Data\\processed\\processed_iris.csv")