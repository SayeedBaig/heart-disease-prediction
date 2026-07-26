import asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.routes.doctor import router as doctor_router
from api.database.connection import engine
from api.database.base import Base

Base.metadata.create_all(bind=engine)

app = FastAPI()
app.include_router(doctor_router)
client = TestClient(app)

login = client.post('/doctors/login', json={
    'email': 'testdoc2@test.com',
    'password': 'password123'
})
token = login.json()['access_token']

me3 = client.get('/doctors/me', headers={'Authorization': 'Bearer \"' + token + '\"'})
print('With quotes:', me3.status_code, me3.json() if me3.status_code != 200 else '')
