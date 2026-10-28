import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("Iris.csv")

if "Id" in data.columns:
    data = data.drop(columns=["Id"])

X = data.drop(columns=["Species"])
y = data["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

joblib.dump(model, "iris_model.pkl")

with open("metrics.json", "w") as file:
    json.dump({"accuracy": accuracy}, file, indent=4)

print(f"Model accuracy: {accuracy:.4f}")
