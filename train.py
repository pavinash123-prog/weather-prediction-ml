import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset
data = pd.read_csv("weather.csv")

# 👉 Make sure your CSV has columns like: temperature, humidity, wind_speed, condition
print("Columns in dataset:", data.columns)

# 2. Split features and target
X = data.drop("condition", axis=1)   # independent variables
y = data["condition"]                # target (label)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)

# 6. Accuracy
print("✅ Model Accuracy:", accuracy_score(y_test, y_pred))

# 7. Example prediction
sample = [[30, 70, 12]]  # temperature=30°C, humidity=70%, wind_speed=12 km/h
print("Predicted condition:", model.predict(sample))
