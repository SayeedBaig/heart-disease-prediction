from pydantic import BaseModel, Field

class ClinicalInput(BaseModel):
    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 54,
                "gender": 2,
                "height": 172,
                "weight": 78,
                "ap_hi": 145,
                "ap_lo": 92,
                "cholesterol": 2,
                "gluc": 1,
                "smoke": 0,
                "alco": 0,
                "active": 1
            }
        }
    }

    age: int = Field(..., ge=1, le=120)
    gender: int = Field(..., ge=1, le=2)

    height: float = Field(..., ge=100, le=250)
    weight: float = Field(..., ge=30, le=300)

    ap_hi: int = Field(..., ge=50, le=300)
    ap_lo: int = Field(..., ge=30, le=200)

    cholesterol: int = Field(..., ge=1, le=3)
    gluc: int = Field(..., ge=1, le=3)

    smoke: int = Field(..., ge=0, le=1)
    alco: int = Field(..., ge=0, le=1)
    active: int = Field(..., ge=0, le=1)