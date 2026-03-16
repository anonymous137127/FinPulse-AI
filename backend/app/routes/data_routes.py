from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
import pandas as pd
from datetime import datetime

from app.auth import require_role
from app.mongodb import users_collection, financial_collection

router = APIRouter()

@router.post("/upload-csv")
def upload_csv(
    file: UploadFile = File(...),
    user: dict = Depends(require_role(["admin","analyst","auditor"]))
):

    current_user = users_collection.find_one({"username": user["sub"]})

    df = pd.read_csv(file.file)

    records = []

    for _, row in df.iterrows():

        records.append({
            "user_id": str(current_user["_id"]),
            "revenue": float(row[0]),
            "expense": float(row[1]),
            "created_at": datetime.utcnow()
        })

    financial_collection.insert_many(records)

    return {"message": "CSV uploaded successfully"}