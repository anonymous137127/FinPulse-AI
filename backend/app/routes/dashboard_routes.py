from fastapi import APIRouter, Depends
from app.auth import require_role
from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime

from app.auth import require_role
from app.mongodb import users_collection, financial_collection
from app.services.hash_service import generate_ml_hash

router = APIRouter()

@router.get("/dashboard-data")
def dashboard_data(
    user: dict = Depends(require_role(["admin","analyst","auditor"]))
):

    return {
        "message": "Dashboard endpoint working"
    }

@router.get("/hash-ml-results")
def hash_ml_results(
    user: dict = Depends(require_role(["admin","analyst","auditor"]))
):

    current_user = users_collection.find_one({"username": user["sub"]})

    if not current_user:
        raise HTTPException(status_code=404, detail="User not found")

    data = list(
        financial_collection.find({
            "user_id": str(current_user["_id"]),
            "type": {"$ne": "forecast_result"}
        })
    )

    if not data:
        return {
            "sha256_hash": None,
            "total_records": 0,
            "high_risk": 0,
            "medium_risk": 0,
            "low_risk": 0
        }

    hash_value, high, medium, low, results = generate_ml_hash(data)

    return {
        "sha256_hash": hash_value,
        "total_records": len(results),
        "high_risk": high,
        "medium_risk": medium,
        "low_risk": low
    }