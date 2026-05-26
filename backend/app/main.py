from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from jose import jwt
from passlib.context import CryptContext

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import joblib
import os

# Machine Learning

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Blockchain

from app.blockchain import Blockchain
import hashlib
import json
from time import time


from app.mongodb import users_collection

import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB

from app.mongodb import (
    db,
    users_collection,
    financial_collection,
    blockchain_collection
)

# PASSWORD HASHING

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

# ENV VARIABLES

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))

security = HTTPBearer()


# ---------------- AUTH ----------------

def create_access_token(data: dict):

    to_encode = data.copy()

    expire = datetime.utcnow() + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(
        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM
    )

    return encoded_jwt


# ---------------- TOKEN VERIFICATION ----------------

def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):

    token = credentials.credentials

    try:

        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        return payload

    except jwt.ExpiredSignatureError:

        raise HTTPException(
            status_code=401,
            detail="Token expired ❌"
        )

    except jwt.JWTError:

        raise HTTPException(
            status_code=401,
            detail="Invalid token ❌"
        )


# ---------------- ROLE AUTHORIZATION ----------------

def require_role(roles: list):

    def role_checker(user: dict = Depends(verify_token)):

        user_role = user.get("role")

        if user_role not in roles:

            raise HTTPException(
                status_code=403,
                detail="Access denied ❌"
            )

        return user

    return role_checker

# FASTAPI APP

app = FastAPI(title="FinPulse API 🚀")

# CORS

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# BASIC ROUTE

@app.get("/")
def home():
    return {"message": "Backend Running Successfully 🚀"}

@app.get("/mongo-test")
def mongo_test():

    data = {"message": "MongoDB working 🚀"}

    users_collection.insert_one(data)

    return {"status": "MongoDB Insert Success"}

# ---------------- REGISTER ----------------

@app.post("/register")
def register(username: str, password: str, role: str):

    username = username.strip()
    password = password.strip()

    # Validate role
    if role not in ["admin", "analyst", "auditor"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid role ❌"
        )

    # Validate username
    if len(username) < 3:
        raise HTTPException(
            status_code=400,
            detail="Username must be at least 3 characters"
        )

    # Validate password
    if len(password) < 4:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 4 characters"
        )

    # Check if user already exists
    existing_user = users_collection.find_one({"username": username})

    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Username already exists ❌"
        )

    # Hash password
    hashed_password = pwd_context.hash(password)

    # Insert user into MongoDB
    users_collection.insert_one({
        "username": username,
        "password": hashed_password,
        "role": role,
        "created_at": datetime.utcnow()
    })

    return {
        "message": "User registered successfully ✅",
        "username": username,
        "role": role
    }

# ---------------- LOGIN ----------------

from app.mongodb import users_collection

@app.post("/login")
def login(username: str, password: str):

    # Clean input
    username = username.strip()
    password = password.strip()

    # Find user in MongoDB
    user = users_collection.find_one({"username": username})

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password ❌"
        )

    # Verify bcrypt password
    if not pwd_context.verify(password, user["password"]):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password ❌"
        )

    # Create JWT token
    token = create_access_token({
        "sub": user["username"],
        "role": user["role"]
    })

    return {
        "message": "Login successful ✅",
        "access_token": token,
        "token_type": "bearer",
        "username": user["username"],
        "role": user["role"]
    }

# ---------------- UPLOAD CSV (FIXED) ----------------

@app.post("/upload-csv")
def upload_csv(
    file: UploadFile = File(...),
    user: dict = Depends(require_role(["admin", "analyst", "auditor"]))
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed ❌")

    current_user = users_collection.find_one({"username": user["sub"]})
    if not current_user:
        raise HTTPException(status_code=404, detail="User not found ❌")

    user_id = str(current_user["_id"])

    try:
        df = pd.read_csv(file.file)

        # ── Clean column names ─────────────────────────────────
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        df = df.replace(r'[\$,₹,]', '', regex=True)

        print("Detected columns:", list(df.columns))

        # ══════════════════════════════════════════════════
        # ✅ FIX 1: SMART COLUMN DETECTION
        # Keyword matching — never picks date/id/quantity
        # ══════════════════════════════════════════════════

        REVENUE_KEYWORDS = ["revenue", "income", "sales", "earnings", "turnover"]
        EXPENSE_KEYWORDS = ["expense", "expenses", "cost", "costs", "spending", "price"]
        SKIP_COLS = ["date", "month", "year", "week", "day", "id",
                     "order_id", "index", "quantity", "qty", "count"]

        def find_col(df, keywords, skip):
            """Find first column whose name contains any keyword."""
            for kw in keywords:
                for col in df.columns:
                    if kw in col and col not in skip:
                        return col
            return None

        revenue_col = find_col(df, REVENUE_KEYWORDS, SKIP_COLS)
        expense_col = find_col(df, EXPENSE_KEYWORDS, SKIP_COLS)

        # Fallback: first two pure numeric cols (excluding skip list)
        if not revenue_col or not expense_col:
            numeric_cols = []
            for col in df.columns:
                if col in SKIP_COLS:
                    continue
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().sum() / max(len(df), 1) > 0.8:
                    numeric_cols.append(col)

            if len(numeric_cols) < 1:
                raise HTTPException(
                    status_code=400,
                    detail="No numeric revenue/expense columns found ❌"
                )

            revenue_col = revenue_col or numeric_cols[0]
            expense_col = expense_col or (
                numeric_cols[1] if len(numeric_cols) > 1 else None
            )

        print(f"Revenue col: {revenue_col}")
        print(f"Expense col: {expense_col}")

        # ══════════════════════════════════════════════════
        # ✅ FIX 2: CONVERT REVENUE & EXPENSE TO NUMERIC
        # ══════════════════════════════════════════════════

        df[revenue_col] = pd.to_numeric(df[revenue_col], errors="coerce")

        if expense_col:
            df[expense_col] = pd.to_numeric(df[expense_col], errors="coerce")
        else:
            df["expense_derived"] = df[revenue_col] * 0.70
            expense_col = "expense_derived"

        # ══════════════════════════════════════════════════
        # ✅ FIX 3: DETECT IF TRANSACTIONAL (needs grouping)
        # ══════════════════════════════════════════════════

        DATE_COL_NAMES = ["date", "order_date", "transaction_date",
                          "invoice_date", "created_at", "month"]

        date_col = next(
            (c for c in df.columns if c in DATE_COL_NAMES), None
        )

        records = []

        if date_col:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df = df.dropna(subset=[date_col])
                df["_month"] = df[date_col].dt.to_period("M")

                monthly_agg = (
                    df.groupby("_month")
                    .agg(
                        revenue=(revenue_col, "sum"),
                        expense=(expense_col, "sum")
                    )
                    .reset_index()
                    .sort_values("_month")
                )

                print(f"Grouped into {len(monthly_agg)} monthly buckets")

                for idx, row in monthly_agg.iterrows():
                    records.append({
                        "user_id": user_id,
                        "revenue": float(row["revenue"]),
                        "expense": float(row["expense"]),
                        "month":   str(row["_month"]),
                        "row_index": idx,
                        "created_at": datetime.utcnow() + timedelta(seconds=idx)
                    })

            except Exception as date_err:
                print("Date parsing failed, falling back to row-by-row:", date_err)
                date_col = None

        if not date_col:
            df = df.dropna(subset=[revenue_col])

            for idx, row in df.iterrows():
                revenue = row.get(revenue_col)
                expense = row.get(expense_col, revenue * 0.70)

                if pd.isna(revenue):
                    continue

                records.append({
                    "user_id":   user_id,
                    "revenue":   float(revenue),
                    "expense":   float(expense) if not pd.isna(expense) else float(revenue) * 0.70,
                    "row_index": int(idx),
                    "created_at": datetime.utcnow() + timedelta(seconds=int(idx))
                })

        if not records:
            raise HTTPException(
                status_code=400,
                detail="No valid rows found in CSV ❌"
            )

        # ── Delete old data & insert new ──────────────────
        financial_collection.delete_many({"user_id": user_id})
        financial_collection.insert_many(records)

        # ── Run ML models ──────────────────────────────────
        try:
            classify_risk_xgb(user)
        except Exception as e:
            print("Risk ML Error:", e)

        try:
            forecast_revenue(user)
        except Exception as e:
            print("Forecast ML Error:", e)

        # ── KPIs ───────────────────────────────────────────
        total_revenue = sum(r["revenue"] for r in records)
        total_expense = sum(r["expense"] for r in records)

        return {
            "message": "CSV uploaded + AI analysis completed ✅",
            "rows_inserted":       len(records),
            "data_type_detected":  "transactional (grouped by month)" if date_col else "pre-aggregated",
            "columns_used": {
                "revenue": revenue_col,
                "expense": expense_col
            },
            "kpis": {
                "total_revenue": round(total_revenue, 2),
                "total_expense": round(total_expense, 2),
                "net_profit":    round(total_revenue - total_expense, 2)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"CSV processing error: {str(e)}"
        )

# ---------------- REVENUE FORECAST (FIXED) ----------------

@app.get("/forecast-revenue")
def forecast_revenue(
    user: dict = Depends(require_role(["admin", "analyst", "auditor"]))
):
    # ── 1. Fetch user ──────────────────────────────────────────
    current_user = users_collection.find_one({"username": user["sub"]})
    if not current_user:
        raise HTTPException(status_code=404, detail="User not found ❌")

    user_id = str(current_user["_id"])

    # ── 2. Fetch only raw financial rows (not forecast results) ─
    data = list(
        financial_collection.find({
            "user_id": user_id,
            "$or": [
                {"type": {"$exists": False}},
                {"type": {"$ne": "forecast_result"}}
            ]
        }).sort("created_at", 1)
    )

    if len(data) < 3:
        return {
            "next_month_prediction": 0,
            "model_accuracy_r2": 0,
            "months_used_for_training": [],
            "message": "Need at least 3 records for forecasting"
        }

    # ── 3. Extract valid revenues ───────────────────────────────
    revenues = []
    for row in data:
        try:
            val = float(row.get("revenue", 0))
            if val > 0:
                revenues.append(val)
        except:
            continue

    if len(revenues) < 3:
        return {
            "next_month_prediction": 0,
            "model_accuracy_r2": 0,
            "months_used_for_training": revenues
        }

    # ── 4. Group into monthly buckets (6–12 months) ─────────────
    num_months = min(12, max(6, len(revenues) // 3))
    chunk_size = max(1, len(revenues) // num_months)

    monthly_totals = []
    for i in range(num_months):
        start = i * chunk_size
        end = (start + chunk_size) if (i < num_months - 1) else len(revenues)
        if start < len(revenues):
            monthly_totals.append(sum(revenues[start:end]))

    monthly_totals = np.array(monthly_totals, dtype=float)

    # ── 5. Create lag features ──────────────────────────────────
    def create_lag_features(series: np.ndarray, n_lags: int):
        X, y = [], []
        for i in range(n_lags, len(series)):
            X.append(series[i - n_lags : i])
            y.append(series[i])
        return np.array(X), np.array(y)

    n_lags = min(3, len(monthly_totals) - 1)
    X, y = create_lag_features(monthly_totals, n_lags)

    if len(X) < 2:
        model = LinearRegression()
        model.fit(X, y)
        accuracy = float(max(0.0, model.score(X, y)))

    else:
        # ── 6. Time-series train/test split (no shuffle) ────────
        split_idx = max(1, int(len(X) * 0.8))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        model = LinearRegression()
        model.fit(X_train, y_train)

        # ── 7. Real R² accuracy ─────────────────────────────────
        if len(X_test) > 0:
            y_pred = model.predict(X_test)
            raw_r2 = r2_score(y_test, y_pred)
            accuracy = float(max(0.0, min(1.0, raw_r2)))
        else:
            accuracy = float(max(0.0, model.score(X_train, y_train)))

    # ── 8. Predict next month ───────────────────────────────────
    last_window = monthly_totals[-n_lags:].reshape(1, -1)
    prediction = float(model.predict(last_window)[0])
    prediction = max(0.0, prediction)

    # ── 9. Save model & result ──────────────────────────────────
    joblib.dump(model, "revenue_model.pkl")

    financial_collection.update_one(
        {"user_id": user_id, "type": "forecast_result"},
        {
            "$set": {
                "prediction": round(prediction, 2),
                "accuracy":   round(accuracy, 4),
                "n_lags":     n_lags,
                "created_at": datetime.utcnow()
            }
        },
        upsert=True
    )

    return {
        "next_month_prediction":    round(prediction, 2),
        "model_accuracy_r2":        round(accuracy, 4),
        "months_used_for_training": monthly_totals.tolist(),
        "lags_used":                n_lags
    }

# ---------------- XGBOOST RISK CLASSIFICATION ----------------

@app.get("/classify-risk-xgb")
def classify_risk_xgb(
    user: dict = Depends(require_role(["admin", "analyst", "auditor"]))
):
    try:
        current_user = users_collection.find_one({"username": user["sub"]})

        if not current_user:
            raise HTTPException(status_code=404, detail="User not found ❌")

        user_id = str(current_user["_id"])

        data = list(financial_collection.find({
            "user_id": user_id
        }))

        if not data:
            return {
                "message": "No financial data uploaded yet",
                "total_records": 0,
                "results": []
            }

        valid_data = []

        for record in data:
            try:
                revenue = float(record.get("revenue", 0))
                valid_data.append({
                    "_id": record["_id"],
                    "revenue": revenue
                })
            except:
                continue

        if len(valid_data) < 2:
            return {
                "message": "Not enough data, assigning default risk",
                "total_records": len(valid_data),
                "results": []
            }

        revenues = np.array(
            [d["revenue"] for d in valid_data]
        ).reshape(-1, 1)

        labels = []
        for r in revenues:
            value = r[0]
            if value < 1000:
                labels.append("Low")
            elif value < 5000:
                labels.append("Medium")
            else:
                labels.append("High")

        if len(valid_data) < 10:
            for i, record in enumerate(valid_data):
                financial_collection.update_one(
                    {"_id": record["_id"]},
                    {"$set": {"risk_level": labels[i]}}
                )

            return {
                "message": "Rule-based risk assigned ✅",
                "total_records": len(valid_data),
                "results": labels
            }

        encoder = LabelEncoder()
        y = encoder.fit_transform(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            revenues,
            y,
            test_size=0.2,
            random_state=42
        )

        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            eval_metric="mlogloss"
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        predictions = model.predict(revenues)

        results = []

        for i, record in enumerate(valid_data):
            risk_label = encoder.inverse_transform([predictions[i]])[0]

            financial_collection.update_one(
                {"_id": record["_id"]},
                {"$set": {"risk_level": risk_label}}
            )

            results.append({
                "id": str(record["_id"]),
                "revenue": record["revenue"],
                "risk": risk_label
            })

        joblib.dump(model, "xgb_risk_model.pkl")

        return {
            "message": "XGBoost risk classification completed ✅",
            "accuracy": round(float(accuracy), 4),
            "total_records": len(valid_data),
            "results": results
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Risk classification failed: {str(e)}"
        )


def run_risk_classification(user):

    current_user = users_collection.find_one({"username": user["sub"]})
    user_id = str(current_user["_id"])

    data = list(financial_collection.find({"user_id": user_id}))

    for record in data:
        revenue = float(record.get("revenue", 0))

        if revenue < 1000:
            risk = "Low"
        elif revenue < 5000:
            risk = "Medium"
        else:
            risk = "High"

        financial_collection.update_one(
            {"_id": record["_id"]},
            {"$set": {"risk_level": risk}}
        )

# ---------------- HASH ML RESULTS ----------------

@app.get("/hash-ml-results")
def hash_ml_results(
    user: dict = Depends(require_role(["admin","analyst","auditor"]))
):

    current_user = users_collection.find_one({"username": user["sub"]})

    if not current_user:
        raise HTTPException(status_code=404, detail="User not found ❌")

    data = list(
        financial_collection.find(
            {"user_id": str(current_user["_id"])}
        )
    )

    if not data:
        return {
            "sha256_hash": None,
            "total_records": 0,
            "high_risk": 0,
            "normal": 0
        }

    results = []
    high = 0
    normal = 0

    for record in data:
        risk = record.get("risk_level", "Normal")

        if "High" in risk:
            high += 1
        else:
            normal += 1

        results.append({
            "id": str(record["_id"]),
            "risk_level": risk
        })

    results_string = json.dumps(results, sort_keys=True)
    result_hash = hashlib.sha256(results_string.encode()).hexdigest()

    return {
        "sha256_hash": result_hash,
        "total_records": len(results),
        "high_risk": high,
        "normal": normal
    }

# ---------------- HASH FINANCIAL DATA ----------------

@app.get("/hash-financial-data")
def hash_financial_data(
    user: dict = Depends(require_role(["admin","analyst","auditor"]))
):

    current_user = users_collection.find_one({"username": user["sub"]})

    if not current_user:
        raise HTTPException(status_code=404, detail="User not found ❌")

    data = list(
        financial_collection.find(
            {"user_id": str(current_user["_id"])}
        )
    )

    if not data:
        return {
            "message": "No financial data uploaded yet",
            "sha256_hash": None,
            "total_records": 0
        }

    dataset = []

    for record in data:
        dataset.append({
            "id": str(record["_id"]),
            "revenue": record["revenue"],
            "expense": record["expense"]
        })

    dataset_string = json.dumps(dataset, sort_keys=True)
    dataset_hash = hashlib.sha256(dataset_string.encode()).hexdigest()

    return {
        "message": "Dataset hashed successfully ✅",
        "sha256_hash": dataset_hash,
        "total_records": len(dataset)
    }

# ---------------- BLOCKCHAIN SYSTEM ----------------

class Blockchain:

    def __init__(self):

        self.chain = list(blockchain_collection.find({}, {"_id": 0}).sort("index", 1))

        if len(self.chain) == 0:
            self.create_genesis_block()

    def calculate_hash(self, index, timestamp, data, previous_hash):

        block_string = json.dumps({
            "index": index,
            "timestamp": timestamp,
            "data": data,
            "previous_hash": previous_hash
        }, sort_keys=True)

        return hashlib.sha256(block_string.encode()).hexdigest()

    def create_genesis_block(self):

        genesis_block = {
            "index": 0,
            "timestamp": str(datetime.utcnow()),
            "data": "Genesis Block",
            "previous_hash": "0"
        }

        genesis_block["current_hash"] = self.calculate_hash(
            genesis_block["index"],
            genesis_block["timestamp"],
            genesis_block["data"],
            genesis_block["previous_hash"]
        )

        self.chain.append(genesis_block)
        blockchain_collection.insert_one(dict(genesis_block))
