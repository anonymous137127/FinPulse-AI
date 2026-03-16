from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
import pandas as pd
import numpy as np
from datetime import datetime

from app.auth import require_role
from app.mongodb import users_collection, financial_collection

router = APIRouter()


@router.post("/upload-csv")
def upload_csv(
    file: UploadFile = File(...),
    user: dict = Depends(require_role(["admin","analyst","auditor"]))
):

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")

    current_user = users_collection.find_one({"username": user["sub"]})

    if not current_user:
        raise HTTPException(status_code=404, detail="User not found")

    try:

        df = pd.read_csv(file.file)

        # -------- CLEAN COLUMN NAMES --------
        df.columns = df.columns.str.strip().str.lower()

        # -------- REMOVE CURRENCY SYMBOLS --------
        df = df.replace(r'[\$,₹]', '', regex=True)

        # -------- DETECT NUMERIC COLUMNS --------
        numeric_cols = []

        for col in df.columns:

            converted = pd.to_numeric(df[col], errors="coerce")

            if converted.notna().sum() > 0:
                df[col] = converted
                numeric_cols.append(col)

        if len(numeric_cols) < 2:
            raise HTTPException(
                status_code=400,
                detail="Dataset must contain at least 2 numeric columns"
            )

        revenue_col = numeric_cols[0]
        expense_col = numeric_cols[1]

        records = []

        for _, row in df.iterrows():

            revenue = row.get(revenue_col)
            expense = row.get(expense_col)

            if pd.isna(revenue) or pd.isna(expense):
                continue

            records.append({

                "user_id": str(current_user["_id"]),

                "revenue": float(revenue),
                "expense": float(expense),

                # store full row for analytics
                "original_data": row.to_dict(),

                "dataset_columns": list(df.columns),

                "created_at": datetime.utcnow()

            })

        if records:
            financial_collection.insert_many(records)

        return {

            "message": "Dataset uploaded successfully",

            "rows_inserted": len(records),

            "detected_revenue_column": revenue_col,
            "detected_expense_column": expense_col,

            "total_columns_detected": len(df.columns)

        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"CSV processing error: {str(e)}"
        )