import uuid

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def _register_and_login_patient():
    email = f"patient-{uuid.uuid4().hex[:12]}@example.com"
    password = "secure-test-password"
    register = client.post(
        "/patients/register",
        json={
            "full_name": "Isolation Test Patient",
            "email": email,
            "password": password,
            "phone": "555-0100",
            "gender": "Female",
            "date_of_birth": "1990-01-01",
        },
    )
    assert register.status_code == 200

    login = client.post("/patients/login", json={"email": email, "password": password})
    assert login.status_code == 200
    return register.json(), {"Authorization": f"Bearer {login.json()['access_token']}"}


def test_patient_cannot_read_another_patients_prediction_history_or_create_their_prediction():
    first_patient, first_headers = _register_and_login_patient()
    second_patient, _ = _register_and_login_patient()

    history = client.get(
        f"/patients/{second_patient['patient_id']}/predictions",
        headers=first_headers,
    )
    assert history.status_code == 403

    prediction = client.post(
        "/predict",
        headers=first_headers,
        json={
            "patient_id": second_patient["patient_id"],
            "age": 36,
            "gender": 1,
            "height": 165,
            "weight": 62,
            "ap_hi": 120,
            "ap_lo": 80,
            "cholesterol": 1,
            "gluc": 1,
            "smoke": 0,
            "alco": 0,
            "active": 1,
        },
    )
    assert prediction.status_code == 403


def test_uploads_require_an_authenticated_account():
    response = client.post("/upload/ecg", files={"file": ("ecg.csv", b"0,1,2", "text/csv")})
    assert response.status_code == 401
