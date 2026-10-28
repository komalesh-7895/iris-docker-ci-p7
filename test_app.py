import app as iris_app

def test_health_endpoint():
    client = iris_app.app.test_client()
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json["status"] == "healthy"

def test_prediction_endpoint(monkeypatch):
    client = iris_app.app.test_client()

    class MockModel:
        def predict(self, values):
            return ["Iris-setosa"]

    monkeypatch.setattr(iris_app, "model", MockModel())

    response = client.post(
        "/predict",
        json={
            "SepalLengthCm": 5.1,
            "SepalWidthCm": 3.5,
            "PetalLengthCm": 1.4,
            "PetalWidthCm": 0.2
        }
    )

    assert response.status_code == 200
    assert response.json["prediction"] == "Iris-setosa"

def test_missing_features():
    client = iris_app.app.test_client()
    response = client.post("/predict", json={"SepalLengthCm": 5.1})

    assert response.status_code == 400
    assert response.json["error"] == "Missing required features"
