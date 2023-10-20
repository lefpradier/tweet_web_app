import sys

sys.path.insert(1, "backend")
from fastapi.testclient import TestClient
from main import app


# fastapi auto request
client = TestClient(app)


# test unitaire de connexion à l'API
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200  # successful http request
    assert response.json() == {"message": "Welcome from the API"}


# test unitaire de prédiction sur un tweet correct
def test_good():
    response = client.post("/tweet", params={"tweet": "i like cats"})
    assert response.status_code == 200
    assert response.json() == {"message": "", "score": 0.19351422786712646}


# test unitaire de prédiction sur un tweet au formatage incorrect
def test_websites():
    response = client.post("/tweet", params={"tweet": "https://www.google.com"})
    assert response.status_code == 400
    assert response.json() == {
        "detail": "Tweet is empty or contains only hypertext links"
    }


# test unitaire de prédiction sur un tweet non anglophone
def test_language():
    response = client.post("/tweet", params={"tweet": "J'adore les croissants"})
    assert response.status_code == 200
    assert response.json() == {
        "score": 0.11344850808382034,
        "message": "Warning: 2 words out of 3 are unknown. Warning : Tweet does not appear to be in English. However, language detection can be unreliable for tweets below 3 words. Detected language: FR with a probability of 0.764",
    }


# test unitaire sur un tweet contenant des mots inconnus
def test_warning():
    response = client.post(
        "/tweet", params={"tweet": "What art thou that usurp’st this time of night"}
    )
    assert response.status_code == 200
    {"score": 0.3702499270439148, "message": "Warning: 1 words out of 6 are unknown. "}
