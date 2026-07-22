import pytest
from fastapi.testclient import TestClient
import uuid
import random
from api.main import app
from api.utils.auth import create_access_token

client = TestClient(app)

@pytest.fixture(scope="module")
def auth_data():
    email = f"testdoctor_{uuid.uuid4().hex[:6]}@example.com"
    register_response = client.post("/doctors/register", json={
        "email": email,
        "password": "testpassword123",
        "full_name": "Test Doctor",
        "specialization": "Cardiology",
        "hospital": "General Hospital",
        "license_number": f"LIC-{random.randint(1000, 9999)}"
    })
    assert register_response.status_code == 201

    login_response = client.post("/doctors/login", json={
        "email": email,
        "password": "testpassword123"
    })
    assert login_response.status_code == 200
    data = login_response.json()
    return f"Bearer {data['access_token']}", data['doctor']['doctor_id']

@pytest.fixture(scope="module")
def headers(auth_data):
    return {"Authorization": auth_data[0]}

@pytest.fixture(scope="module")
def doctor_id(auth_data):
    return auth_data[1]

def test_unauthorized_access():
    response = client.get("/patients/")
    assert response.status_code == 401
    
def test_invalid_jwt():
    response = client.get("/patients/", headers={"Authorization": "Bearer invalid_token_123"})
    assert response.status_code == 401


def test_invalid_doctor_subject_is_rejected_before_database_query():
    token = create_access_token({"sub": "16", "role": "doctor"})
    response = client.get(
        "/doctors/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401

def test_patient_crud(headers):
    # Create patient
    email = f"patient_{uuid.uuid4().hex[:6]}@example.com"
    payload = {
        "full_name": "John Doe",
        "email": email,
        "password": "testpassword123",
        "phone": "555-0000-000",
        "gender": "Male",
        "date_of_birth": "1980-01-01"
    }
    
    # 1. POST
    create_res = client.post("/patients/register", json=payload)
    assert create_res.status_code == 200
    patient_id = create_res.json()["id"]

    login_res = client.post(
        "/patients/login",
        json={"email": email, "password": payload["password"]},
    )
    assert login_res.status_code == 200
    profile_res = client.get(
        "/patients/me",
        headers={"Authorization": f"Bearer {login_res.json()['access_token']}"},
    )
    assert profile_res.status_code == 200
    assert profile_res.json()["patient_id"] == create_res.json()["patient_id"]
    
    # 2. GET
    get_res = client.get(f"/patients/{patient_id}", headers=headers)
    assert get_res.status_code == 200
    
    # 3. List
    list_res = client.get("/patients/", headers=headers)
    assert list_res.status_code == 200
    
    # 4. Search
    search_res = client.get(f"/patients/search?q={email}", headers=headers)
    assert search_res.status_code == 200
    assert len(search_res.json()) > 0
    
    # 5. Update
    update_res = client.put(f"/patients/{patient_id}", json={"full_name": "John Updated"}, headers=headers)
    assert update_res.status_code == 200

def test_full_workflow(headers, doctor_id):
    # A. Create Patient
    email = f"patient_{uuid.uuid4().hex[:6]}@example.com"
    payload = {
        "full_name": "Jane Doe",
        "email": email,
        "password": "testpassword123",
        "phone": "555-1111-111",
        "gender": "Female",
        "date_of_birth": "1990-01-01"
    }
    p_res = client.post("/patients/register", json=payload)
    assert p_res.status_code == 200
    p_data = p_res.json()
    p_id = p_data.get("id") or p_data.get("patient_id")

    # B. Create Appointment
    app_payload = {
        "patient_id": p_id,
        "doctor_id": doctor_id,
        "preferred_date": "2030-01-01",
        "preferred_time": "10:00:00",
        "symptoms": "Chest pain",
        "reason": "Checkup"
    }
    app_res = client.post("/appointments/", json=app_payload, headers=headers)
    assert app_res.status_code == 201
    appointment_id = app_res.json()["appointment_id"]
    
    # Approve appointment
    appr_res = client.put(f"/appointments/{appointment_id}/approve", headers=headers)
    assert appr_res.status_code == 200
    
    # C. Create Diagnosis
    diag_payload = {
        "appointment_id": appointment_id,
        "clinical_data": {
            "gender": 1,
            "height": 165,
            "weight": 65,
            "ap_hi": 120,
            "ap_lo": 80,
            "cholesterol": 1,
            "gluc": 1,
            "smoke": 0,
            "alco": 0,
            "active": 1
        }
    }
    diag_res = client.post("/diagnosis", json=diag_payload, headers=headers)
    assert diag_res.status_code == 200
    diagnosis_id = diag_res.json()["diagnosis_id"]
    
    # D. Run Prediction
    pred_res = client.post("/prediction", json={"diagnosis_id": diagnosis_id}, headers=headers)
    assert pred_res.status_code == 200
    prediction_id = pred_res.json()["prediction_id"]

    completed_appointment = client.get(
        f"/appointments/{appointment_id}", headers=headers
    )
    assert completed_appointment.status_code == 200
    assert completed_appointment.json()["status"] == "Completed"
    
    # E. Add Doctor Notes
    note_res = client.post("/notes/", json={
        "diagnosis_id": diagnosis_id,
        "notes": "Patient looks healthy",
        "prescription": "Aspirin",
        "advice": "Rest",
        "follow_up": "In 6 months"
    }, headers=headers)
    assert note_res.status_code == 200
    
    # F. Reports
    doc_rep = client.get(f"/reports/{prediction_id}/doctor", headers=headers)
    assert doc_rep.status_code == 200
    pat_rep = client.get(f"/reports/{prediction_id}/patient", headers=headers)
    assert pat_rep.status_code == 200
    
    # G. History
    hist_res = client.get(f"/history/patient/{p_res.json().get('id', p_id)}", headers=headers)
    assert hist_res.status_code == 200
    assert len(hist_res.json()) > 0
    
    # H. 404 Case
    not_found = client.get("/diagnosis/00000000-0000-0000-0000-000000000000", headers=headers)
    assert not_found.status_code == 404
    
    # I. 422 Case
    invalid_payload = client.post("/appointments/", json={"patient_id": "abc"}, headers=headers)
    assert invalid_payload.status_code == 422

def test_database_rollback_case(headers):
    # Try creating diagnosis for invalid appointment ID -> should rollback and return 400
    diag_payload = {
        "appointment_id": "00000000-0000-0000-0000-000000000000",
        "clinical_data": {
            "gender": 1, "height": 165, "weight": 65, "ap_hi": 120, "ap_lo": 80,
            "cholesterol": 1, "gluc": 1, "smoke": 0, "alco": 0, "active": 1
        }
    }
    res = client.post("/diagnosis", json=diag_payload, headers=headers)
    assert res.status_code in (400, 404)
