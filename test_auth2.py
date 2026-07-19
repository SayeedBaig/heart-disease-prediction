
import asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.routes.appointment import router as appointment_router
from api.database.connection import engine
from api.database.base import Base

Base.metadata.create_all(bind=engine)

app = FastAPI()
app.include_router(appointment_router)
client = TestClient(app)

import uuid
from api.utils.auth import create_access_token
token = create_access_token({'sub': str(uuid.uuid4()), 'email': 'test@test.com'})

res = client.post('/appointments/', headers={'Authorization': f'Bearer {token}'}, json={
    'patient_id': 1,
    'doctor_id': str(uuid.uuid4()),
    'preferred_date': '2030-01-01',
    'preferred_time': '10:00:00',
    'symptoms': 'Chest pain',
    'reason': 'Checkup'
})
print('Response:', res.status_code, res.json() if res.status_code != 200 else '')
