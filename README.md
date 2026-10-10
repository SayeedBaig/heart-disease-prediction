# CardioAI: Heart Disease Prediction & Digital Twin Platform

CardioAI is an advanced medical platform designed to predict, simulate, and analyze heart disease risks using cutting-edge AI, Retrieval-Augmented Generation (RAG), and Digital Twin technology. It serves as a comprehensive tool for both patients and healthcare professionals to monitor cardiovascular health and make data-driven medical decisions.

## 🚀 Key Features

*   **Dual Portal System:** Dedicated interfaces for both Patients and Doctors with role-specific dashboards.
*   **AI Health Assistant:** Integrated RAG-powered chatbot to answer patient queries using trusted medical knowledge bases (ACC/AHA/ESC guidelines).
*   **Digital Twin Engine:** Simulates a patient's cardiovascular state to predict how different lifestyle changes (e.g., diet, exercise, medication) might impact their heart health over time.
*   **Clinical Data Analysis:** Support for uploading and processing clinical data, including ECG and Echo findings, for comprehensive risk assessment.
*   **Comprehensive Reports:** Generates detailed, easy-to-understand reports for patients and highly technical clinical summaries for doctors.

## 🛠️ Technology Stack

### Frontend
*   **React (Vite):** Fast, modern UI development.
*   **Modern UI/UX:** Built with a highly responsive, modern, and beautiful design featuring custom CSS animations, dynamic layouts, and immersive 3D elements.

### Backend
*   **FastAPI:** High-performance Python backend for serving API requests.
*   **RAG Pipeline:** Utilizes FAISS vector stores and LLM models to generate medically accurate responses based on embedded clinical guidelines.
*   **Digital Twin Simulator:** Python-based simulation engine for scenario testing (Blood Pressure, Cholesterol, Diabetes, etc.).

## 📁 Project Structure

*   `/frontend`: Contains the React/Vite web application.
*   `/api`: FastAPI routes, models, and core API logic.
*   `/digital_twin`: The logic and models for the cardiovascular simulation engine.
*   `/rag`: Retrieval-Augmented Generation pipeline, including vector stores, embeddings, and prompt generators.
*   `/reports`: Automated generation of patient and doctor reports.

## 🚦 Getting Started

### Prerequisites
*   Node.js (v18+)
*   Python (3.9+)

### Running the Backend
1. Navigate to the root directory.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/Scripts/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your `.env` file with necessary API keys (e.g., `GROQ_API_KEY`).
5. Run the FastAPI server:
   ```bash
   uvicorn api.main:app --reload
   ```

### Running the Frontend
1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```

## 👥 Branching Strategy
*   `main`: Stable, deployable code.
*   `develop`: Primary integration branch for new features.
*   `feature/*`: Branches for developing new features.
