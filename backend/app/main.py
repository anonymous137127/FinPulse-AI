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
import json
import hashlib

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

# MongoDB

from app.mongodb import (
    db,
    users_collection,
    financial_collection,
    blockchain_collection
)

# Schemas

from app.schemas import RegisterRequest

# PASSWORD HASHING


pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

# JWT CONFIG


SECRET_KEY = "my_super_secret_key_123"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()

# ===============================
# FASTAPI APP
# ===============================

app = FastAPI(title="FinPulse API 🚀")

# CORS

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AUTH FUNCTIONS

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(
            credentials.credentials,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )
        return payload
    except:
        raise HTTPException(status_code=401, detail="Invalid token ❌")


def require_role(roles: list):

    def role_checker(user: dict = Depends(verify_token)):

        if user.get("role") not in roles:
            raise HTTPException(
                status_code=403,
                detail="Access denied ❌"
            )

        return user

    return role_checker

# BASIC ROUTE

@app.get("/")
def home():
    return {"message": "Backend Running Successfully 🚀"}

# ---------------- AUTH ----------------

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except:
        raise HTTPException(status_code=401, detail="Invalid token ❌")

def require_role(roles: list):
    def role_checker(user: dict = Depends(verify_token)):

        if user.get("role") not in roles:
            raise HTTPException(
                status_code=403,
                detail="Access denied ❌"
            )

        return user

    return role_checker
# ---------------- BASIC ROUTES ----------------

@app.get("/")
def home():
    return {"message": "Backend Running Successfully 🚀"}


# ---------------- MONGO TEST ----------------

from app.mongodb import users_collection

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
# ---------------- UNIVERSAL CSV ETL (MongoDB Version) ----------------

@app.post("/upload-csv")
def upload_csv(
    file: UploadFile = File(...),
    user: dict = Depends(require_role(["admin","analyst","auditor"]))
):

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed ❌")

    # get current user from MongoDB
    current_user = users_collection.find_one({"username": user["sub"]})

    if not current_user:
        raise HTTPException(status_code=404, detail="User not found ❌")

    # delete previous user data
    financial_collection.delete_many({
        "user_id": str(current_user["_id"])
    })

    rows_inserted = 0

    try:

        df = pd.read_csv(file.file)

        # clean column names
        df.columns = df.columns.str.strip().str.lower()

        # remove currency symbols
        df = df.replace(r'[\$,₹]', '', regex=True)

        # detect numeric columns automatically
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
                "created_at": datetime.utcnow()
            })

        if records:
            financial_collection.insert_many(records)

        rows_inserted = len(records)

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"CSV processing error: {str(e)}"
        )

    return {
        "message": "CSV uploaded successfully ✅",
        "rows_inserted": rows_inserted
    }

# ---------------- REVENUE FORECAST ----------------

@app.get("/forecast-revenue")
def forecast_revenue(
    user: dict = Depends(require_role(["admin","analyst","auditor"]))
):

    # get logged in user from MongoDB
    current_user = users_collection.find_one({"username": user["sub"]})

    if not current_user:
        raise HTTPException(status_code=404, detail="User not found ❌")

    # get user's financial data
    data = list(
        financial_collection.find(
            {"user_id": str(current_user["_id"])}
        )
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
            revenues.append(float(row["revenue"]))
        except:
            continue

    if len(revenues) < 5:
        return {
            "next_month_prediction": 0,
            "model_accuracy_r2": 0,
            "months_used_for_training": revenues
        }

    # ---------------- GROUP INTO 6 MONTHS ----------------

    chunk_size = max(1, len(revenues) // 6)

    monthly_totals = []

    for i in range(6):
        start = i * chunk_size
        end = start + chunk_size
        monthly_totals.append(sum(revenues[start:end]))

    monthly_totals = np.array(monthly_totals)

    # ---------------- TRAIN MODEL ----------------

    X = np.arange(len(monthly_totals)).reshape(-1, 1)
    y = monthly_totals

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    # ---------------- EVALUATE ----------------

    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test, y_pred)

    # ---------------- PREDICT NEXT MONTH ----------------

    next_index = np.array([[len(monthly_totals)]])
    prediction = model.predict(next_index)[0]

    # save model
    joblib.dump(model, "revenue_model.pkl")

    # store prediction in MongoDB
    financial_collection.insert_one({
        "type": "forecast_result",
        "user_id": str(current_user["_id"]),
        "prediction": float(round(prediction, 2)),
        "accuracy": float(round(accuracy, 4)),
        "created_at": datetime.utcnow()
    })

    return {
        "next_month_prediction": round(float(prediction), 2),
        "model_accuracy_r2": round(float(accuracy), 4),
        "months_used_for_training": monthly_totals.tolist()
    }
# ---------------- ANOMALY DETECTION ----------------

@app.get("/detect-anomalies")
def detect_anomalies(
    user: dict = Depends(require_role(["admin","analyst","auditor"]))
):

    # get logged-in user
    current_user = users_collection.find_one({"username": user["sub"]})

    if not current_user:
        raise HTTPException(status_code=404, detail="User not found ❌")

    # load user's financial data
    data = list(
        financial_collection.find(
            {"user_id": str(current_user["_id"])}
        )
    )

    if not data:
        return {
            "message": "No financial data uploaded yet",
            "total_records": 0,
            "anomalies_found": 0,
            "results": []
        }

    if len(data) < 10:
        return {
            "message": "Upload more data for anomaly detection",
            "total_records": len(data),
            "anomalies_found": 0,
            "results": []
        }

    revenues = np.array(
        [float(d["revenue"]) for d in data]
    ).reshape(-1, 1)

    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(revenues)

    predictions = model.predict(revenues)
    scores = model.decision_function(revenues)

    results = []

    for i, record in enumerate(data):

        risk = "High Risk 🚨" if predictions[i] == -1 else "Normal"

        # update MongoDB record
        financial_collection.update_one(
            {"_id": record["_id"]},
            {"$set": {"risk_level": risk}}
        )

        results.append({
            "id": str(record["_id"]),
            "revenue": record["revenue"],
            "risk_level": risk,
            "anomaly_score": round(float(scores[i]), 4)
        })

    return {
        "total_records": len(data),
        "anomalies_found": int(sum(predictions == -1)),
        "results": results
    }
# ---------------- RISK CLASSIFICATION ----------------

@app.get("/classify-risk")
def classify_risk(
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
            "total_records": 0,
            "results": []
        }

    if len(data) < 10:
        return {
            "message": "Need at least 10 records for classification",
            "total_records": len(data),
            "results": []
        }

    revenues = np.array(
        [float(d["revenue"]) for d in data]
    ).reshape(-1, 1)

    # ---------- Rule-based labels ----------
    labels = []

    for value in revenues:
        v = value[0]

        if v < 1000:
            labels.append("Low")
        elif v < 5000:
            labels.append("Medium")
        else:
            labels.append("High")

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    model = LogisticRegression()
    model.fit(revenues, y)

    predictions = model.predict(revenues)

    results = []

    for i, record in enumerate(data):

        risk_label = encoder.inverse_transform([predictions[i]])[0]

        financial_collection.update_one(
            {"_id": record["_id"]},
            {"$set": {"risk_level": risk_label}}
        )

        results.append({
            "id": str(record["_id"]),
            "revenue": record["revenue"],
            "classified_risk": risk_label
        })

    return {
        "message": "Risk classification completed ✅",
        "total_records": len(data),
        "results": results
    }
# ---------------- XGBOOST RISK CLASSIFICATION ----------------

@app.get("/classify-risk-xgb")
def classify_risk_xgb(
    user: dict = Depends(require_role(["admin","analyst","auditor"]))
):

    # get logged-in user
    current_user = users_collection.find_one({"username": user["sub"]})

    if not current_user:
        raise HTTPException(status_code=404, detail="User not found ❌")

    # get financial data
    data = list(
        financial_collection.find(
            {"user_id": str(current_user["_id"])}
        )
    )

    if not data:
        return {
            "message": "No financial data uploaded yet",
            "total_records": 0,
            "results": []
        }

    if len(data) < 10:
        return {
            "message": "Need at least 10 records for XGBoost classification",
            "total_records": len(data),
            "results": []
        }

    revenues = np.array(
        [float(d["revenue"]) for d in data]
    ).reshape(-1, 1)

    # --------- create labels ----------
    labels = []

    for value in revenues:
        v = value[0]

        if v < 1000:
            labels.append("Low")
        elif v < 5000:
            labels.append("Medium")
        else:
            labels.append("High")

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        revenues, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # save model
    joblib.dump(model, "xgb_risk_model.pkl")
    joblib.dump(encoder, "risk_label_encoder.pkl")

    all_predictions = model.predict(revenues)
    probabilities = model.predict_proba(revenues)

    results = []

    for i, record in enumerate(data):

        risk_label = encoder.inverse_transform([all_predictions[i]])[0]
        confidence = round(float(max(probabilities[i]) * 100), 2)

        # update MongoDB
        financial_collection.update_one(
            {"_id": record["_id"]},
            {"$set": {"risk_level": risk_label}}
        )

        results.append({
            "id": str(record["_id"]),
            "revenue": record["revenue"],
            "classified_risk": risk_label,
            "confidence_percent": confidence
        })

    return {
        "message": "XGBoost risk classification completed ✅",
        "model_accuracy": round(float(accuracy), 4),
        "total_records": len(data),
        "model_saved": True,
        "results": results
    }
# ---------------- FORECAST API ----------------

@app.get("/forecast")
def forecast(
    user: dict = Depends(require_role(["admin","analyst","auditor"]))
):

    if not os.path.exists("revenue_model.pkl"):
        raise HTTPException(
            status_code=400,
            detail="Forecast model not trained yet ❌"
        )

    model = joblib.load("revenue_model.pkl")

    # get logged-in user
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
            "next_month_prediction": None
        }

    if len(data) < 3:
        return {
            "message": "Need at least 3 records for forecasting",
            "next_month_prediction": None
        }

    next_index = np.array([[len(data)]])
    prediction = model.predict(next_index)[0]

    return {
        "message": "Forecast generated successfully ✅",
        "next_month_prediction": round(float(prediction), 2)
    }
@app.post("/detect-risk")
def detect_risk(
    revenue: float,
    user: dict = Depends(require_role(["admin","analyst","auditor"]))
):

    if not os.path.exists("xgb_risk_model.pkl"):
        raise HTTPException(
            status_code=400,
            detail="Risk model not trained yet ❌"
        )

    if not os.path.exists("risk_label_encoder.pkl"):
        raise HTTPException(
            status_code=400,
            detail="Label encoder not found ❌"
        )

    model = joblib.load("xgb_risk_model.pkl")
    encoder = joblib.load("risk_label_encoder.pkl")

    input_data = np.array([[revenue]])

    prediction = model.predict(input_data)[0]
    risk = encoder.inverse_transform([prediction])[0]

    return {
        "message": "Risk prediction successful ✅",
        "revenue": revenue,
        "predicted_risk": risk
    }
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

## ---------------- BLOCKCHAIN SYSTEM ----------------

class Blockchain:

    def __init__(self):

        # load chain from MongoDB
        self.chain = list(blockchain_collection.find({}, {"_id": 0}).sort("index", 1))

        # create genesis block ONLY if blockchain empty
        if len(self.chain) == 0:
            self.create_genesis_block()

    # ---------------- HASH FUNCTION ----------------
    def calculate_hash(self, index, timestamp, data, previous_hash):

        block_string = json.dumps({
            "index": index,
            "timestamp": timestamp,
            "data": data,
            "previous_hash": previous_hash
        }, sort_keys=True)

        return hashlib.sha256(block_string.encode()).hexdigest()

    # ---------------- GENESIS BLOCK ----------------
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

        blockchain_collection.insert_one(genesis_block)

    # ---------------- ADD BLOCK ----------------
    def add_block(self, data):

        previous_block = self.chain[-1]

        new_block = {
            "index": len(self.chain),
            "timestamp": str(datetime.utcnow()),
            "data": data,
            "previous_hash": previous_block["current_hash"]
        }

        new_block["current_hash"] = self.calculate_hash(
            new_block["index"],
            new_block["timestamp"],
            new_block["data"],
            new_block["previous_hash"]
        )

        self.chain.append(new_block)

        blockchain_collection.insert_one(new_block)

        return new_block

    # ---------------- INTEGRITY CHECK ----------------
    def is_chain_valid(self):

        if len(self.chain) <= 1:
            return True

        for i in range(1, len(self.chain)):

            current = self.chain[i]
            previous = self.chain[i - 1]

            recalculated_hash = self.calculate_hash(
                current["index"],
                current["timestamp"],
                current["data"],
                current["previous_hash"]
            )

            if current["current_hash"] != recalculated_hash:
                return False

            if current["previous_hash"] != previous["current_hash"]:
                return False

        return True


# 🔥 Initialize Blockchain
blockchain = Blockchain()

# ---------------- LOAD BLOCKCHAIN FROM MONGODB ----------------

def load_blockchain_from_db():

    try:

        blocks = list(
            blockchain_collection.find({}, {"_id": 0}).sort("index", 1)
        )

        blockchain.chain = []

        # If DB empty → create genesis block
        if len(blocks) == 0:
            blockchain.create_genesis_block()
            print("Genesis block created")
            return

        for block in blocks:

            # safety check for required fields
            if not all(k in block for k in ["index", "timestamp", "data", "previous_hash", "current_hash"]):
                continue

            blockchain.chain.append({
                "index": block["index"],
                "timestamp": block["timestamp"],
                "data": block["data"],
                "previous_hash": block["previous_hash"],
                "current_hash": block["current_hash"]
            })

        print(f"Blockchain loaded successfully ({len(blockchain.chain)} blocks)")

    except Exception as e:
        print("Blockchain loading error:", str(e))

# ---------------- STARTUP EVENT ----------------

@app.on_event("startup")
def startup_event():

    print("Loading blockchain from MongoDB...")

    load_blockchain_from_db()

    if blockchain.is_chain_valid():
        print("Blockchain integrity verified ✅")
    else:
        print("Blockchain integrity FAILED ❌")

# =========================================
# ADD BLOCK API
# =========================================

@app.post("/add-block")
def add_block(user: dict = Depends(require_role(["admin","analyst","auditor"]))):

    current_user = users_collection.find_one({"username": user["sub"]})

    if not current_user:
        raise HTTPException(status_code=404, detail="User not found ❌")

    data = list(financial_collection.find({
        "user_id": str(current_user["_id"])
    }))

    if not data:
        return {
            "message": "No financial data uploaded yet",
            "block_created": False
        }

    dataset = []

    for record in data:
        dataset.append({
            "id": str(record["_id"]),
            "revenue": record["revenue"],
            "expense": record["expense"],
            "risk_level": record.get("risk_level", "Normal")
        })

    new_block = blockchain.add_block(dataset)

    return {
        "message": "Block added successfully ✅",
        "block_index": new_block["index"],
        "timestamp": new_block["timestamp"],
        "previous_hash": new_block["previous_hash"],
        "current_hash": new_block["current_hash"]
    }

# =========================================
# VIEW BLOCKCHAIN
# =========================================

@app.get("/view-chain")
def view_chain(user: dict = Depends(require_role(["admin","analyst","auditor"]))):

    try:

        if not blockchain.chain:
            return {
                "message": "Blockchain is empty",
                "length": 0,
                "is_valid": True,
                "chain": []
            }

        return {
            "message": "Blockchain retrieved successfully",
            "length": len(blockchain.chain),
            "is_valid": blockchain.is_chain_valid(),
            "chain": blockchain.chain
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Blockchain error: {str(e)}"
        )


# =========================================
# VERIFY BLOCKCHAIN
# =========================================

@app.get("/verify-integrity")
def verify_integrity(user: dict = Depends(require_role(["admin","analyst","auditor"]))):

    try:

        valid = blockchain.is_chain_valid()

        if valid:
            return {
                "status": "Valid",
                "icon": "✅",
                "message": "Blockchain integrity verified successfully",
                "total_blocks": len(blockchain.chain)
            }

        return {
            "status": "Tampered",
            "icon": "❌",
            "message": "Blockchain has been modified",
            "total_blocks": len(blockchain.chain)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Integrity verification error: {str(e)}"
        )

# =========================================
# KPI API
# =========================================

@app.get("/kpis")
def get_kpis(user: dict = Depends(require_role(["admin","analyst","auditor"]))):

    current_user = users_collection.find_one({"username": user["sub"]})

    if not current_user:
        raise HTTPException(status_code=404, detail="User not found")

    data = list(financial_collection.find({
        "user_id": str(current_user["_id"])
    }))

    total_revenue = 0
    total_expense = 0

    for row in data:

        try:
            total_revenue += float(row["revenue"])
        except:
            pass

        try:
            total_expense += float(row["expense"])
        except:
            pass

    return {
        "total_revenue": total_revenue,
        "total_expense": total_expense,
        "net_profit": total_revenue - total_expense
    }


# =========================================
# REVENUE FORECAST (GRAPH DATA)
# =========================================

@app.get("/revenue-forecast")
def revenue_forecast(user: dict = Depends(require_role(["admin","analyst","auditor"]))):

    current_user = users_collection.find_one({"username": user["sub"]})

    if not current_user:
        raise HTTPException(status_code=404, detail="User not found")

    data = list(financial_collection.find({
        "user_id": str(current_user["_id"])
    }))

    if not data:
        return []

    today = datetime.today()

    months = [
        (today - relativedelta(months=i)).strftime("%b")
        for i in range(5, -1, -1)
    ]

    revenues = []

    for row in data:
        try:
            revenues.append(float(row["revenue"]))
        except:
            continue

    if not revenues:
        return []

    chunk_size = max(1, len(revenues)//6)

    forecast = []

    for i, month in enumerate(months):

        start = i * chunk_size
        end = start + chunk_size

        forecast.append({
            "month": month,
            "revenue": sum(revenues[start:end])
        })

    return forecast


# =========================================
# CHART DATA
# =========================================

@app.get("/chart-data")
def chart_data(user: dict = Depends(require_role(["admin","analyst","auditor"]))):

    current_user = users_collection.find_one({"username": user["sub"]})

    data = list(financial_collection.find({
        "user_id": str(current_user["_id"])
    }))

    today = datetime.today()

    months = [
        (today - relativedelta(months=i)).strftime("%b")
        for i in range(5, -1, -1)
    ]

    revenues = [float(r["revenue"]) for r in data]
    expenses = [float(r["expense"]) for r in data]

    chunk_size = max(1, len(revenues)//6)

    result = []

    for i, m in enumerate(months):

        start = i * chunk_size
        end = start + chunk_size

        result.append({
            "month": m,
            "revenue": sum(revenues[start:end]),
            "expense": sum(expenses[start:end])
        })

    return result


# =========================================
# DASHBOARD DATA
# =========================================

@app.get("/dashboard-data")
def get_dashboard_data(user: dict = Depends(require_role(["admin","analyst","auditor"]))):

    kpis = get_kpis(user)

    forecast = revenue_forecast(user)

    chart = chart_data(user)

    prediction = forecast_revenue(user)

    anomaly = classify_risk_xgb(user)

    risk_records = hash_ml_results(user)

    blockchain_status = verify_integrity(user)

    return {
        "kpis": kpis,
        "forecast": forecast,
        "prediction": prediction,
        "chart": chart,
        "anomaly": anomaly,
        "risk_records": risk_records,
        "blockchain": blockchain_status
    }

@app.get("/reset-blockchain")
def reset_blockchain():

    blockchain_collection.delete_many({})

    blockchain.chain = []

    blockchain.create_genesis_block()

    return {
        "message": "Blockchain reset successfully",
        "status": "Valid"
    }

class Blockchain:

    def __init__(self):

        self.chain = list(
            blockchain_collection.find({}, {"_id":0}).sort("index",1)
        )

        if len(self.chain) == 0:
            self.create_genesis_block()