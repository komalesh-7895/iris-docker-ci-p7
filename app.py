from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("iris_model.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "service": "iris-prediction"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "JSON data is required"}), 400

    features = [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]

    if not all(feature in data for feature in features):
        return jsonify({"error": "Missing required features"}), 400

    try:
        values = [[float(data[feature]) for feature in features]]
        prediction = model.predict(values)[0]
        return jsonify({"prediction": str(prediction)})
    except (TypeError, ValueError):
        return jsonify({"error": "Feature values must be numeric"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
