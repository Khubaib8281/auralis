from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/v1/voice/score")
    assert response.status_code in [200, 400]

def test_exception_case():
    with open("tests/invalid.txt", "rb") as f:
        response = client.post("api/v1/voice/score", files = {"file", :f})
    assert response.status_code == 400