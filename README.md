# CardioAI – Multimodal Heart Disease Prediction System

A cutting‑edge AI platform that assesses cardiovascular risk by integrating clinical data, ECG signals, and echocardiography images.

CardioAI fuses modality‑specific predictions using an intelligent feature‑fusion engine and provides explainable AI insights, a RAG‑powered medical assistant, and a Digital Twin simulation for patient‑specific scenario analysis.

The platform extends beyond prediction through patient and doctor portals, a Retrieval-Augmented Generation (RAG) based medical assistant, Digital Twin simulation, prediction history, and appointment management.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Prediction Workflow](#prediction-workflow)
- [Technology Stack](#technology-stack)
- [Application Modules](#application-modules)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Team Responsibilities](#team-responsibilities)
- [Future Scope](#future-scope)
- [Disclaimer](#disclaimer)

## Overview
Traditional heart disease prediction approaches may rely on a single data modality or provide limited context around their predictions. CardioAI explores a multimodal approach by integrating different sources of cardiovascular information into a unified prediction workflow.

The system processes available patient information through clinical, ECG, and echocardiography modules. Their outputs are standardized and combined through an intelligent feature fusion mechanism to produce an overall cardiovascular risk assessment.

The resulting information is made accessible through a web application designed for patients and healthcare professionals.

### Project Objectives
- Integrate clinical, ECG, and echocardiography data for cardiovascular risk assessment.
- Combine modality-specific prediction outputs using intelligent feature fusion.
- Present risk level, risk percentage, and confidence information.
- Provide supporting medical information and contextual explanations.
- Enable patient-specific risk analysis through Digital Twin simulation.
- Support prediction history, patient reports, and appointment workflows.
- Provide dedicated interfaces for patients and doctors.

## Key Features

### 1. Multimodal Heart Disease Prediction
- Clinical data analysis.
- ECG signal analysis.
- Echocardiography analysis.
- Unified cardiovascular risk assessment.

### 2. Intelligent Feature Fusion
- Standardization of outputs from individual prediction modules.
- Weighted combination of modality-specific results.
- Confidence-aware handling of available model outputs.
- Generation of an overall risk assessment.

### 3. Explainable AI and RAG Assistant
- Retrieval of relevant medical information from a knowledge base.
- Contextual responses to healthcare-related questions.
- Supporting information to help users understand prediction results.

### 4. Digital Twin Simulation
- Patient-specific cardiovascular risk analysis.
- Interactive modification of selected health parameters.
- Scenario-based exploration of potential changes in predicted risk.

### 5. Patient Portal
- Patient registration and authentication.
- Clinical information submission and prediction access.
- Prediction history and patient reports.
- Digital Twin and AI assistant access.
- Appointment booking and profile management.

### 6. Doctor Portal
- Doctor registration and authentication.
- Access to available patient records and prediction information.
- Patient management and appointment workflows.
- Access to supported clinical information and records.

### 7. Report Generation
- Structured presentation of prediction results.
- Historical prediction records.
- Patient-oriented report generation and download.

### 8. Healthcare Workflow Integration
- Role-based application access.
- Appointment scheduling and management.
- Persistent storage of relevant application data.
- Frontend-backend integration through REST APIs.

## System Architecture
CardioAI follows a modular architecture that separates the user interface, backend services, prediction pipeline, and data storage.

**Main architectural layers:**
- **Presentation Layer:** React-based web interface for patients and doctors.
- **API and Service Layer:** FastAPI endpoints for authentication, prediction, reports, appointments, and supporting services.
- **Prediction Layer:** Clinical, ECG, and echocardiography models.
- **Fusion Layer:** Standardization and combination of modality-specific prediction outputs.
- **Supporting Intelligence Layer:** RAG-based medical information retrieval and Digital Twin simulation.
- **Data Layer:** PostgreSQL for structured application data and supporting storage for medical knowledge retrieval.

The modular structure supports integration of the individual components within a unified healthcare application.

## Prediction Workflow
The primary CardioAI workflow follows these stages:

1. **Patient Input:** Collect available clinical information and relevant ECG and echocardiography inputs.
2. **Input Validation:** Validate incoming information and apply the appropriate preprocessing steps.
3. **Modality-Specific Processing:** Process the available inputs through their respective prediction modules.
4. **Output Standardization:** Convert module outputs into a consistent representation for downstream processing.
5. **Intelligent Feature Fusion:** Combine available outputs using the fusion mechanism.
6. **Risk Assessment:** Generate the overall risk result and associated assessment information.
7. **Result Presentation:** Display the result through the application and make supported reporting and explanation features available.

The RAG assistant, Digital Twin, reports, and appointment workflows complement the primary prediction pipeline.

## Technology Stack

| Component | Technologies |
|---|---|
| Frontend | React, Vite, JavaScript, Tailwind CSS |
| Backend | Python, FastAPI, Uvicorn |
| API Communication | REST APIs, Axios |
| Database | PostgreSQL |
| Database ORM and Migrations | SQLAlchemy, Alembic |
| Machine Learning | Scikit-learn, Pandas, NumPy |
| Deep Learning | PyTorch, TensorFlow/Keras |
| Medical Knowledge Retrieval | RAG, FAISS |
| Visualization and Reporting | Interactive charts, PDF report generation |
| Development Tools | Git, GitHub, VS Code |

The table describes the technologies used across the project. Exact dependencies and versions should be verified against the current project configuration.

## Application Modules

| Module | Purpose |
|---|---|
| Clinical Prediction | Analyses available clinical parameters. |
| ECG Prediction | Processes ECG signals for cardiovascular assessment. |
| Echocardiography Prediction | Analyses available echocardiography inputs. |
| Fusion Engine | Combines standardized modality-specific outputs. |
| RAG Assistant | Retrieves relevant medical information for contextual responses. |
| Digital Twin | Supports patient-specific scenario analysis. |
| Authentication | Manages patient and doctor access. |
| Reports | Stores and presents prediction-related reports. |
| Appointments | Supports appointment booking and management. |
| Prediction History | Enables retrieval of previous prediction records. |

## Project Structure
The following is a logical overview of the main application areas. The exact folder names may differ in the current repository.

```text
CardioAI/
├── frontend/
│   └── src/
│       ├── components/
│       ├── pages/
│       ├── services/
│       └── ...
├── api/
│   ├── main.py
│   ├── routes/
│   ├── schemas/
│   ├── services/
│   └── ...
├── clinical_module/
├── ecg_module/
├── echo_module/
├── fusion_module/
├── digital_twin/
├── reports/
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites
Install the following before running the application:
- Python compatible with the project's dependencies.
- Node.js and npm.
- PostgreSQL.
- Git.

### 1. Clone the Repository
```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd CardioAI
```
*Replace the placeholder with your actual repository URL.*

### 2. Set Up the Backend
Create and activate a Python virtual environment:
```bash
python -m venv .venv
```
Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```
Install the backend dependencies:
```bash
pip install -r requirements.txt
```
Configure the required environment variables and PostgreSQL connection before starting the backend.
Apply schema migrations to the configured database before starting the updated API:
```bash
alembic upgrade head
```

Run the FastAPI application from the project root if `api/main.py` is the configured entry point:
```bash
uvicorn api.main:app --reload
```
When running successfully, the interactive API documentation is generally available at:
`http://127.0.0.1:8000/docs`

### 3. Set Up the Frontend
Open a separate terminal:
```bash
cd frontend
npm install
npm run dev
```
Vite will display the local development URL in the terminal, typically:
`http://localhost:5173`

### 4. Configure the Database
- Create the PostgreSQL database required by the application.
- Configure the database connection using the project's environment variables.
- Apply reviewed schema migrations with `alembic upgrade head` before starting the API. Migrations are not run automatically.
- Ensure the backend can connect to the database before testing the application.

*Important: Follow the actual configuration files and existing migration setup in the repository. The commands above are a starting point, not a substitute for project-specific configuration.*


## Configuration
Create the environment files expected by the application and configure the required values locally.
Typical configuration categories include:
- PostgreSQL connection URL.
- Frontend API base URL.
- Authentication and token settings.
- Medical knowledge base or vector-store paths.
- Upload and report storage paths.
- Any required AI provider configuration.

> **Note:** Do not commit database passwords, secret keys, access tokens, private credentials, or other sensitive configuration values to GitHub. Use the project's existing environment variable names rather than introducing new names without updating the application.

## Team Responsibilities
CardioAI is developed as a collaborative major project.

| Team Member | Primary Responsibility |
|---|---|
| **Sayeed Baig** | Backend architecture, FastAPI services, database integration, prediction workflow, feature fusion, API integration, and overall system integration |
| **Sowmya** | Frontend implementation, UI/UX, application pages, and frontend integration |
| **Rishitha** | Clinical data module, Digital Twin, Literature review, research background, related work, research objectives, and literature comparison |
| **Akash** | RAG based medical assistant, Echocardiography module, Dataset preparation, preprocessing, feature engineering, experimental setup, evaluation metrics, and results analysis |

*Responsibilities are summarized from the team's project work distribution.*

## Future Scope
Potential areas for further development include:
- Validation using larger and more diverse clinical datasets.
- Evaluation of modality-specific and fused prediction performance.
- Improved interpretability of multimodal predictions.
- Integration with clinical information systems and medical devices.
- Further evaluation of patient-specific simulation approaches.
- Security, privacy, and deployment improvements for real-world environments.

## Disclaimer
CardioAI is an academic research and software development project. Its predictions, explanations, and simulations are intended for research and educational purposes and must not be treated as a confirmed medical diagnosis.

The system is not a substitute for professional medical evaluation, clinical judgment, or emergency care. Clinical use would require appropriate validation, safety assessment, and regulatory review.

## Acknowledgement
Developed as a Computer Science and Engineering major project at RV Institute of Technology and Management, Bengaluru, exploring multimodal machine learning and AI-assisted cardiovascular risk assessment.

*CardioAI — Bringing multimodal AI and healthcare workflows together.*
