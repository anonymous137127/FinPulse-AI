from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime
import joblib

from app.auth import require_role
from app.mongodb import users_collection, financial_collection

from app.services.forecast_service import forecast_revenue_logic
from app.services.risk_service import classify_risk_xgb_logic

router = APIRouter()


# ---------------- FORECAST REVENUE ----------------

@router.get("/forecast-revenue")
def forecast_revenue(
    user: dict = Depends(require_role(["admin","analyst","auditor"]))
):

    current_user = users_collection.find_one({"username": user["sub"]})

    if not current_user:
        raise HTTPException(status_code=404, detail="User not found ❌")

    data = list(
        financial_collection.find({
            "user_id": str(current_user["_id"]),
            "type": {"$ne": "forecast_result"}
        })
    )

    if len(data) < 5:
        return {
            "next_month_prediction": 0,
            "model_accuracy_r2": 0,
            "months_used_for_training": []
        }

    revenues = []

    for row in data:
        try:
            revenues.append(float(row.get("revenue",0)))
        except:
            continue

    if len(revenues) < 5:
        return {
            "next_month_prediction": 0,
            "model_accuracy_r2": 0,
            "months_used_for_training": revenues
        }

    prediction, accuracy, months_used, model = forecast_revenue_logic(revenues)

    joblib.dump(model, "revenue_model.pkl")

    financial_collection.insert_one({
        "type": "forecast_result",
        "user_id": str(current_user["_id"]),
        "prediction": float(round(prediction,2)),
        "accuracy": float(round(accuracy,4)),
        "created_at": datetime.utcnow()
    })

    return {
        "next_month_prediction": round(float(prediction),2),
        "model_accuracy_r2": round(float(accuracy),4),
        "months_used_for_training": months_used
    }

@router.get("/classify-risk-xgb")
def classify_risk_xgb(
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

    if len(data) < 10:
        return {
            "message": "Need at least 10 records",
            "results": []
        }

    features = []
    valid_data = []

    for record in data:

        try:

            revenue = float(record.get("revenue",0))
            expense = float(record.get("expense",0))

            profit = revenue - expense

            if revenue == 0:
                profit_margin = 0
            else:
                profit_margin = profit / revenue

            features.append([revenue, expense, profit, profit_margin])
            valid_data.append(record)

        except:
            continue

    import numpy as np
    X = np.array(features)

    # -------- Create Risk Labels (Business Logic) --------

    labels = []

    for f in features:

        revenue, expense, profit, margin = f

        if profit < 0:
            labels.append("High")

        elif margin < 0.1:
            labels.append("Medium")

        else:
            labels.append("Low")

    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()

    y = encoder.fit_transform(labels)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss"
    )

    model.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    joblib.dump(model,"xgb_risk_model.pkl")
    joblib.dump(encoder,"risk_label_encoder.pkl")

    results = []

    for i, record in enumerate(valid_data):

        risk_label = encoder.inverse_transform([predictions[i]])[0]

        confidence = round(float(max(probabilities[i]) * 100),2)

        financial_collection.update_one(
            {"_id": record["_id"]},
            {"$set": {"risk_level": risk_label}}
        )

        results.append({
            "id": str(record["_id"]),
            "revenue": record.get("revenue"),
            "expense": record.get("expense"),
            "classified_risk": risk_label,
            "confidence_percent": confidence
        })

    return {
        "message": "Risk classification completed",
        "model_accuracy": round(float(accuracy),4),
        "total_records": len(results),
        "results": results
    }