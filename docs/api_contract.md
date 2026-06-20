# Phase-2 API Contract

## Health Check

GET /health

Response:

```json
{
  "status": "healthy",
  "models_loaded": true
}
```

---

## Heart Disease Prediction

POST /predict

Response:

```json
{
  "report_id": "uuid",

  "clinical": {},
  "ecg": {},
  "echo": {},

  "fusion": {
    "risk_level": "",
    "risk_percentage": 0
  },

  "rag": {
    "summary": "",
    "recommendations": []
  },

  "digital_twin": {
    "current_risk": 0,
    "projected_risk": []
  }
}
```

---

## Digital Twin Simulation

POST /simulate

Response:

```json
{
  "no_change": [],
  "exercise": [],
  "quit_smoking": [],
  "statin": []
}
```

---

## Prediction History

GET /history

Response:

```json
[
  {
    "report_id": "",
    "timestamp": ""
  }
]
```

---

## Report Download

GET /reports/{id}

Response:

PDF File

```
```
