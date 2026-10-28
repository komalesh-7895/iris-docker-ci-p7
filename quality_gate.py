import json
import sys

with open("metrics.json", "r") as file:
    metrics = json.load(file)

accuracy = metrics["accuracy"]

print(f"Model accuracy: {accuracy:.2%}")

if accuracy < 0.85:
    print("QUALITY GATE FAILED: Accuracy is below 85%.")
    sys.exit(1)

print("QUALITY GATE PASSED.")
