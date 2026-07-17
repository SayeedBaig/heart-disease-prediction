# CardioAI Backend API Contract

## Authentication
CardioAI uses Bearer token (JWT) authentication for all protected endpoints. Obtain a token by calling the `/doctors/login` endpoint and include it in the header as: `Authorization: Bearer <token>`.

## Endpoints

### `GET` /health
**Summary**: Health Check
**Description**: 
**Authentication Required**: No

**Responses**:
- **200**: Successful Response

---

### `POST` /predict
**Summary**: Predict heart disease risk
**Description**: Runs the complete CardioAI prediction pipeline using clinical data, ECG, and Echocardiography inputs.
**Authentication Required**: No

**Request Body**:
```json
{
  "age": "integer",
  "gender": "integer",
  "height": "number",
  "weight": "number",
  "ap_hi": "integer",
  "ap_lo": "integer",
  "cholesterol": "integer",
  "gluc": "integer",
  "smoke": "integer",
  "alco": "integer",
  "active": "integer",
  "ecg_path": "string | null",
  "echo_path": "string | null",
  "patient_id": "string | null"
}
```

**Responses**:
- **200**: Prediction completed successfully.
- **422**: Validation Error

---

### `GET` /reports/{prediction_id}/doctor
**Summary**: Generate Doctor Report
**Description**: Generates a detailed doctor report from a prediction.
**Authentication Required**: No

**Path Parameters**:
- `prediction_id` (required: True)

**Responses**:
- **200**: Doctor report generated successfully.
- **422**: Validation Error

---

### `GET` /reports/{prediction_id}/patient
**Summary**: Generate Patient Report
**Description**: Generates a simplified patient-facing report from a prediction.
**Authentication Required**: No

**Path Parameters**:
- `prediction_id` (required: True)

**Responses**:
- **200**: Patient report generated successfully.
- **422**: Validation Error

---

### `GET` /reports/{prediction_id}/doctor/pdf
**Summary**: Generate Doctor PDF
**Description**: Generates and downloads a detailed doctor report PDF for the given prediction.
**Authentication Required**: No

**Path Parameters**:
- `prediction_id` (required: True)

**Responses**:
- **200**: Doctor PDF generated successfully.
- **422**: Validation Error

---

### `GET` /reports/{prediction_id}/patient/pdf
**Summary**: Generate Patient PDF
**Description**: Generates and downloads a patient-facing report PDF for the given prediction.
**Authentication Required**: No

**Path Parameters**:
- `prediction_id` (required: True)

**Responses**:
- **200**: Patient PDF generated successfully.
- **422**: Validation Error

---

### `POST` /reports/{prediction_id}/patient/email
**Summary**: Email Patient Report
**Description**: Generates a patient PDF report and emails it to the registered patient.
**Authentication Required**: No

**Path Parameters**:
- `prediction_id` (required: True)

**Responses**:
- **200**: Email sent successfully.
- **422**: Validation Error

---

### `POST` /reports/{prediction_id}/doctor/email
**Summary**: Email Doctor Report
**Description**: Generates a doctor PDF report and emails it to the registered patient.
**Authentication Required**: No

**Path Parameters**:
- `prediction_id` (required: True)

**Responses**:
- **200**: Email sent successfully.
- **422**: Validation Error

---

### `POST` /upload/ecg
**Summary**: Upload ECG file
**Description**: Uploads an ECG file for heart disease prediction.
**Authentication Required**: No

**Request Body**:
```json
{
  "file": "string"
}
```

**Responses**:
- **200**: ECG uploaded successfully.
- **422**: Validation Error

---

### `POST` /upload/echo
**Summary**: Upload Echocardiography file
**Description**: Uploads an echocardiography video for heart disease prediction.
**Authentication Required**: No

**Request Body**:
```json
{
  "file": "string"
}
```

**Responses**:
- **200**: Echo uploaded successfully.
- **422**: Validation Error

---

### `POST` /patients/register
**Summary**: Register a new patient
**Description**: Registers a new patient in the CardioAI system and returns a unique patient ID.
**Authentication Required**: No

**Request Body**:
```json
{
  "full_name": "string",
  "email": "string",
  "phone": "string",
  "gender": "string",
  "date_of_birth": "string"
}
```

**Responses**:
- **200**: Patient registered successfully.
  *Example Response:*
  ```json
  {
    "id": "integer",
    "patient_id": "string",
    "full_name": "string",
    "email": "string",
    "message": "string",
    "created_at": "string"
  }
  ```
- **422**: Validation Error

---

### `GET` /patients/search
**Summary**: Search patients
**Description**: Search patients by name, email, or patient ID.
**Authentication Required**: Yes

**Query Parameters**:
- `q` (required: True)

**Responses**:
- **200**: Matching patients retrieved.
  *Example Response:*
  ```json
  [
    {
        "id": "integer",
        "patient_id": "string",
        "full_name": "string",
        "email": "string",
        "phone": "string",
        "gender": "string",
        "date_of_birth": "string",
        "created_at": "string",
        "updated_at": "string"
    }
  ]
  ```
- **422**: Validation Error

---

### `GET` /patients/
**Summary**: List all patients
**Description**: Returns a paginated list of all registered patients.
**Authentication Required**: Yes

**Query Parameters**:
- `skip` (required: False)
- `limit` (required: False)

**Responses**:
- **200**: Patients list retrieved.
  *Example Response:*
  ```json
  [
    {
        "id": "integer",
        "patient_id": "string",
        "full_name": "string",
        "email": "string",
        "phone": "string",
        "gender": "string",
        "date_of_birth": "string",
        "created_at": "string",
        "updated_at": "string"
    }
  ]
  ```
- **422**: Validation Error

---

### `GET` /patients/{patient_id}
**Summary**: Get patient by ID
**Description**: Returns a single patient by their integer primary key.
**Authentication Required**: Yes

**Path Parameters**:
- `patient_id` (required: True)

**Responses**:
- **200**: Patient retrieved.
  *Example Response:*
  ```json
  {
    "id": "integer",
    "patient_id": "string",
    "full_name": "string",
    "email": "string",
    "phone": "string",
    "gender": "string",
    "date_of_birth": "string",
    "created_at": "string",
    "updated_at": "string"
  }
  ```
- **422**: Validation Error

---

### `PUT` /patients/{patient_id}
**Summary**: Update patient
**Description**: Updates profile fields for an existing patient.
**Authentication Required**: Yes

**Path Parameters**:
- `patient_id` (required: True)

**Request Body**:
```json
{
  "full_name": "string | null",
  "phone": "string | null",
  "gender": "string | null"
}
```

**Responses**:
- **200**: Patient updated.
  *Example Response:*
  ```json
  {
    "id": "integer",
    "patient_id": "string",
    "full_name": "string",
    "email": "string",
    "phone": "string",
    "gender": "string",
    "date_of_birth": "string",
    "created_at": "string",
    "updated_at": "string"
  }
  ```
- **422**: Validation Error

---

### `DELETE` /patients/{patient_id}
**Summary**: Delete patient
**Description**: Permanently removes a patient record.
**Authentication Required**: Yes

**Path Parameters**:
- `patient_id` (required: True)

**Responses**:
- **204**: Patient deleted.
- **422**: Validation Error

---

### `GET` /patients/{patient_id}/predictions
**Summary**: Get prediction history
**Description**: Returns all past predictions for the given patient ID.
**Authentication Required**: No

**Path Parameters**:
- `patient_id` (required: True)

**Responses**:
- **200**: Prediction history retrieved successfully.
  *Example Response:*
  ```json
  {
    "patient_id": "string",
    "total_predictions": "integer",
    "predictions": "array"
  }
  ```
- **422**: Validation Error

---

### `POST` /doctors/register
**Summary**: Register a new doctor
**Description**: Creates a new doctor account with hashed password.
**Authentication Required**: No

**Request Body**:
```json
{
  "full_name": "string",
  "specialization": "string",
  "hospital": "string",
  "email": "string",
  "password": "string"
}
```

**Responses**:
- **201**: Doctor registered successfully.
  *Example Response:*
  ```json
  {
    "doctor_id": "string",
    "full_name": "string",
    "email": "string",
    "message": "string"
  }
  ```
- **422**: Validation Error

---

### `POST` /doctors/login
**Summary**: Doctor login
**Description**: Authenticates a doctor and returns a JWT access token.
**Authentication Required**: No

**Request Body**:
```json
{
  "email": "string",
  "password": "string"
}
```

**Responses**:
- **200**: Login successful.
  *Example Response:*
  ```json
  {
    "access_token": "string",
    "token_type": "string",
    "doctor": "string"
  }
  ```
- **422**: Validation Error

---

### `GET` /doctors/me
**Summary**: Get current doctor profile
**Description**: Returns the profile of the authenticated doctor.
**Authentication Required**: Yes

**Responses**:
- **200**: Doctor profile retrieved.
  *Example Response:*
  ```json
  {
    "doctor_id": "string",
    "full_name": "string",
    "specialization": "string",
    "hospital": "string",
    "email": "string",
    "created_at": "string"
  }
  ```

---

### `PUT` /doctors/me
**Summary**: Update current doctor profile
**Description**: Updates profile fields for the authenticated doctor.
**Authentication Required**: Yes

**Request Body**:
```json
{
  "full_name": "string | null",
  "specialization": "string | null",
  "hospital": "string | null"
}
```

**Responses**:
- **200**: Doctor profile updated.
  *Example Response:*
  ```json
  {
    "doctor_id": "string",
    "full_name": "string",
    "specialization": "string",
    "hospital": "string",
    "email": "string",
    "created_at": "string"
  }
  ```
- **422**: Validation Error

---

### `GET` /doctors/
**Summary**: List all doctors
**Description**: Returns a paginated list of registered doctors.
**Authentication Required**: No

**Query Parameters**:
- `skip` (required: False)
- `limit` (required: False)

**Responses**:
- **200**: Doctors list retrieved.
  *Example Response:*
  ```json
  [
    {
        "doctor_id": "string",
        "full_name": "string",
        "specialization": "string",
        "hospital": "string",
        "email": "string",
        "created_at": "string"
    }
  ]
  ```
- **422**: Validation Error

---

### `POST` /appointments/
**Summary**: Book a new appointment
**Description**: Creates a new appointment record for a patient with a doctor.
**Authentication Required**: Yes

**Request Body**:
```json
{
  "patient_id": "integer",
  "doctor_id": "string",
  "preferred_date": "string",
  "preferred_time": "string",
  "symptoms": "string | null",
  "reason": "string | null"
}
```

**Responses**:
- **201**: Appointment created successfully.
  *Example Response:*
  ```json
  {
    "appointment_id": "string",
    "patient_id": "integer",
    "doctor_id": "string",
    "preferred_date": "string",
    "preferred_time": "string",
    "symptoms": "string | null",
    "reason": "string | null",
    "status": "string",
    "created_at": "string"
  }
  ```
- **422**: Validation Error

---

### `GET` /appointments/
**Summary**: List all appointments
**Description**: Returns a paginated list of all appointments.
**Authentication Required**: Yes

**Query Parameters**:
- `skip` (required: False)
- `limit` (required: False)

**Responses**:
- **200**: Appointments list retrieved.
  *Example Response:*
  ```json
  [
    {
        "appointment_id": "string",
        "patient_id": "integer",
        "doctor_id": "string",
        "preferred_date": "string",
        "preferred_time": "string",
        "symptoms": "string | null",
        "reason": "string | null",
        "status": "string",
        "created_at": "string"
    }
  ]
  ```
- **422**: Validation Error

---

### `GET` /appointments/{appointment_id}
**Summary**: Get appointment by ID
**Description**: Returns a single appointment by its UUID.
**Authentication Required**: Yes

**Path Parameters**:
- `appointment_id` (required: True)

**Responses**:
- **200**: Appointment retrieved.
  *Example Response:*
  ```json
  {
    "appointment_id": "string",
    "patient_id": "integer",
    "doctor_id": "string",
    "preferred_date": "string",
    "preferred_time": "string",
    "symptoms": "string | null",
    "reason": "string | null",
    "status": "string",
    "created_at": "string"
  }
  ```
- **422**: Validation Error

---

### `DELETE` /appointments/{appointment_id}
**Summary**: Delete an appointment
**Description**: Permanently removes an appointment record.
**Authentication Required**: Yes

**Path Parameters**:
- `appointment_id` (required: True)

**Responses**:
- **204**: Appointment deleted.
- **422**: Validation Error

---

### `GET` /appointments/patient/{patient_id}
**Summary**: Get appointments by patient
**Description**: Returns all appointments for a specific patient.
**Authentication Required**: Yes

**Path Parameters**:
- `patient_id` (required: True)

**Responses**:
- **200**: Patient appointments retrieved.
  *Example Response:*
  ```json
  [
    {
        "appointment_id": "string",
        "patient_id": "integer",
        "doctor_id": "string",
        "preferred_date": "string",
        "preferred_time": "string",
        "symptoms": "string | null",
        "reason": "string | null",
        "status": "string",
        "created_at": "string"
    }
  ]
  ```
- **422**: Validation Error

---

### `GET` /appointments/doctor/{doctor_id}
**Summary**: Get appointments by doctor
**Description**: Returns all appointments for a specific doctor.
**Authentication Required**: Yes

**Path Parameters**:
- `doctor_id` (required: True)

**Responses**:
- **200**: Doctor appointments retrieved.
  *Example Response:*
  ```json
  [
    {
        "appointment_id": "string",
        "patient_id": "integer",
        "doctor_id": "string",
        "preferred_date": "string",
        "preferred_time": "string",
        "symptoms": "string | null",
        "reason": "string | null",
        "status": "string",
        "created_at": "string"
    }
  ]
  ```
- **422**: Validation Error

---

### `PUT` /appointments/{appointment_id}/approve
**Summary**: Approve an appointment
**Description**: Sets the appointment status to Approved.
**Authentication Required**: Yes

**Path Parameters**:
- `appointment_id` (required: True)

**Responses**:
- **200**: Appointment approved.
  *Example Response:*
  ```json
  {
    "appointment_id": "string",
    "patient_id": "integer",
    "doctor_id": "string",
    "preferred_date": "string",
    "preferred_time": "string",
    "symptoms": "string | null",
    "reason": "string | null",
    "status": "string",
    "created_at": "string"
  }
  ```
- **422**: Validation Error

---

### `PUT` /appointments/{appointment_id}/reject
**Summary**: Reject an appointment
**Description**: Sets the appointment status to Rejected.
**Authentication Required**: Yes

**Path Parameters**:
- `appointment_id` (required: True)

**Responses**:
- **200**: Appointment rejected.
  *Example Response:*
  ```json
  {
    "appointment_id": "string",
    "patient_id": "integer",
    "doctor_id": "string",
    "preferred_date": "string",
    "preferred_time": "string",
    "symptoms": "string | null",
    "reason": "string | null",
    "status": "string",
    "created_at": "string"
  }
  ```
- **422**: Validation Error

---

### `PUT` /appointments/{appointment_id}/complete
**Summary**: Complete an appointment
**Description**: Sets the appointment status to Completed.
**Authentication Required**: Yes

**Path Parameters**:
- `appointment_id` (required: True)

**Responses**:
- **200**: Appointment completed.
  *Example Response:*
  ```json
  {
    "appointment_id": "string",
    "patient_id": "integer",
    "doctor_id": "string",
    "preferred_date": "string",
    "preferred_time": "string",
    "symptoms": "string | null",
    "reason": "string | null",
    "status": "string",
    "created_at": "string"
  }
  ```
- **422**: Validation Error

---

### `POST` /diagnosis
**Summary**: Create a new diagnosis
**Description**: Creates a diagnosis record associated with an appointment.
**Authentication Required**: Yes

**Request Body**:
```json
{
  "appointment_id": "string",
  "clinical_data": "string",
  "ecg_path": "string | null",
  "echo_path": "string | null"
}
```

**Responses**:
- **200**: Successful Response
  *Example Response:*
  ```json
  {
    "diagnosis_id": "string",
    "appointment_id": "string",
    "patient_id": "integer",
    "doctor_id": "string",
    "clinical_data": "object",
    "ecg_path": "string | null",
    "echo_path": "string | null",
    "status": "string",
    "prediction_id": "integer | null",
    "created_at": "string"
  }
  ```
- **422**: Validation Error

---

### `GET` /diagnosis/{diagnosis_id}
**Summary**: Get diagnosis details
**Description**: Fetches a diagnosis by ID.
**Authentication Required**: Yes

**Path Parameters**:
- `diagnosis_id` (required: True)

**Responses**:
- **200**: Successful Response
  *Example Response:*
  ```json
  {
    "diagnosis_id": "string",
    "appointment_id": "string",
    "patient_id": "integer",
    "doctor_id": "string",
    "clinical_data": "object",
    "ecg_path": "string | null",
    "echo_path": "string | null",
    "status": "string",
    "prediction_id": "integer | null",
    "created_at": "string"
  }
  ```
- **422**: Validation Error

---

### `POST` /prediction
**Summary**: Run prediction on a diagnosis
**Description**: Invokes the prediction pipeline for a given diagnosis.
**Authentication Required**: Yes

**Request Body**:
```json
{
  "diagnosis_id": "string"
}
```

**Responses**:
- **200**: Successful Response
- **422**: Validation Error

---

### `GET` /prediction/{prediction_id}
**Summary**: Get prediction details
**Description**: Fetches standard prediction results by prediction ID.
**Authentication Required**: Yes

**Path Parameters**:
- `prediction_id` (required: True)

**Responses**:
- **200**: Successful Response
- **422**: Validation Error

---

### `GET` /history/patient/{patient_id}
**Summary**: Get patient prediction history
**Description**: 
**Authentication Required**: Yes

**Path Parameters**:
- `patient_id` (required: True)

**Responses**:
- **200**: Successful Response
- **422**: Validation Error

---

### `GET` /history/patient/{patient_id}/compare
**Summary**: Compare latest vs previous prediction
**Description**: 
**Authentication Required**: Yes

**Path Parameters**:
- `patient_id` (required: True)

**Responses**:
- **200**: Successful Response
- **422**: Validation Error

---

### `GET` /history/prediction/{prediction_id}
**Summary**: Get prediction details
**Description**: 
**Authentication Required**: Yes

**Path Parameters**:
- `prediction_id` (required: True)

**Responses**:
- **200**: Successful Response
- **422**: Validation Error

---

### `POST` /notes/
**Summary**: Add a doctor note
**Description**: 
**Authentication Required**: Yes

**Request Body**:
```json
{
  "diagnosis_id": "string",
  "notes": "string | null",
  "prescription": "string | null",
  "advice": "string | null",
  "follow_up": "string | null"
}
```

**Responses**:
- **200**: Successful Response
  *Example Response:*
  ```json
  {
    "note_id": "string",
    "diagnosis_id": "string",
    "notes": "string | null",
    "prescription": "string | null",
    "advice": "string | null",
    "follow_up": "string | null",
    "created_at": "string",
    "updated_at": "string"
  }
  ```
- **422**: Validation Error

---

### `GET` /notes/diagnosis/{diagnosis_id}
**Summary**: Get notes for diagnosis
**Description**: 
**Authentication Required**: Yes

**Path Parameters**:
- `diagnosis_id` (required: True)

**Responses**:
- **200**: Successful Response
  *Example Response:*
  ```json
  [
    {
        "note_id": "string",
        "diagnosis_id": "string",
        "notes": "string | null",
        "prescription": "string | null",
        "advice": "string | null",
        "follow_up": "string | null",
        "created_at": "string",
        "updated_at": "string"
    }
  ]
  ```
- **422**: Validation Error

---

### `GET` /
**Summary**: Root
**Description**: 
**Authentication Required**: No

**Responses**:
- **200**: Successful Response

---

