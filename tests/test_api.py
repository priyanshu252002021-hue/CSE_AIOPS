
# Import TestClient to send simulated HTTP requests to FastAPI
from fastapi.testclient import TestClient

# Import the application object defined in main.py
from main import app

# Create a reusable test client for all test functions
client = TestClient(app)


def test_health_endpoint():
    """
    Verify that the application is running and
    can respond to a health-check request.
    """

    response = client.get("/health")

    # A successful HTTP request should return status code 200
    assert response.status_code == 200

    # Check the response body
    assert response.json() == {"status": "healthy"}


def test_valid_iris_prediction():
    """
    Verify that valid flower measurements produce a prediction.
    """

    payload = {
        "sepal_length": 3.4,
        "sepal_width": 1.4,
        "petal_length": 3.4,
        "petal_width": 0.4
    }

    # Send a simulated HTTP POST request with JSON input
    response = client.post("/predict", json=payload)

    # A valid request should return status code 200
    assert response.status_code == 200

    # Convert JSON response into Python dictionary
    result = response.json()

    # Check the response fields
    assert "prediction" in result
    assert "class_name" in result

    # The predicted class must be one of the three Iris classes
    assert result["class_name"] in [
        "setosa",
        "versicolor",
        "virginica"
    ]

    # The predicted integer must be between 0 and 2
    assert result["prediction"] in [0, 1, 2]


def test_invalid_iris_prediction():
    """
    Verify that negative measurements are rejected.
    """

    payload = {
        "sepal_length": -3.4,
        "sepal_width": 1.4,
        "petal_length": 3.4,
        "petal_width": 0.4
    }

    response = client.post("/predict", json=payload)

    # FastAPI validation error
    assert response.status_code == 422
