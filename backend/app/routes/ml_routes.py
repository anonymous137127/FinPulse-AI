from fastapi import APIRouter, Depends
from app.auth import require_role

router = APIRouter()

@router.get("/forecast-revenue")
def forecast_revenue(
    user: dict = Depends(require_role(["admin","analyst","auditor"]))
):
    return {"prediction": 0}


@router.get("/classify-risk-xgb")
def classify_risk_xgb(
    user: dict = Depends(require_role(["admin","analyst","auditor"]))
):
    return {"message": "Risk classification working"}