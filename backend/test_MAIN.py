import sys

sys.path.insert(1, "backend")
from fastapi.testclient import TestClient
from main import app


# fastapi auto request
client = TestClient(app)


# unit test on auto request : ping-pong
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200  # successful http request
    assert response.json() == {"message": "Welcome from the API"}


# unit test : good item
def test_good():
    response = client.post("/tweet", params={"tweet": "i like cats"})
    assert response.status_code == 200
    assert response.json() == {"message": "", "score": 0.19351422786712646}


# unit test : bad items
def test_websites():
    response = client.post("/tweet", params={"tweet": "https://www.google.com"})
    assert response.status_code == 400
    assert response.json() == {
        "detail": "Tweet is empty or contains only hypertext links"
    }


# unit test : bad language
def test_language():
    response = client.post("/tweet", params={"tweet": "J'adore les croissants"})
    assert response.status_code == 400
    assert response.json() == {"detail": "Tweet does not appear to be in English"}


# unit test : good item but warning
def test_warning():
    response = client.post(
        "/tweet", params={"tweet": "What art thou that usurp’st this time of night"}
    )
    assert response.status_code == 200
    assert response.json() == {
        "message": "Warning: 1 words out of 6 are unknown",
        "score": 0.3702499270439148,
    }
