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

        # Keywords to identify revenue column
        REVENUE_KEYWORDS = ["revenue", "income", "sales", "earnings", "turnover"]
        # Keywords to identify expense/cost column
        EXPENSE_KEYWORDS = ["expense", "expenses", "cost", "costs", "spending", "price"]
        # Columns to always skip (never treat as revenue/expense)
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
                # Only accept if >80% values are valid numbers
                if converted.notna().sum() / max(len(df), 1) > 0.8:
                    numeric_cols.append(col)

            if len(numeric_cols) < 1:
                raise HTTPException(
                    status_code=400,
                    detail="No numeric revenue/expense columns found ❌"
                )

            revenue_col = revenue_col or numeric_cols[0]
            # If no expense col, derive it (80% of revenue as proxy)
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
            # No expense column: derive as 70% of revenue (cost estimate)
            df["expense_derived"] = df[revenue_col] * 0.70
            expense_col = "expense_derived"

        # ══════════════════════════════════════════════════
        # ✅ FIX 3: DETECT IF TRANSACTIONAL (needs grouping)
        # If dataset has a Date column → group by month
        # ══════════════════════════════════════════════════

        DATE_COL_NAMES = ["date", "order_date", "transaction_date",
                          "invoice_date", "created_at", "month"]

        date_col = next(
            (c for c in df.columns if c in DATE_COL_NAMES), None
        )

        records = []

        if date_col:
            # ── Transactional data: aggregate by month ─────
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
                    .sort_values("_month")   # chronological order ✅
                )

                print(f"Grouped into {len(monthly_agg)} monthly buckets")

                # Each row = 1 month, with correct created_at sequence
                for idx, row in monthly_agg.iterrows():
                    records.append({
                        "user_id": user_id,
                        "revenue": float(row["revenue"]),
                        "expense": float(row["expense"]),
                        "month":   str(row["_month"]),   # e.g. "2026-01"
                        "row_index": idx,
                        # Unique timestamps preserve order
                        "created_at": datetime.utcnow() + timedelta(seconds=idx)
                    })

            except Exception as date_err:
                print("Date parsing failed, falling back to row-by-row:", date_err)
                date_col = None   # fall through to row-by-row below

        if not date_col:
            # ── Already aggregated data (e.g. monthly CSV) ─
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
@app.get("/forecast-revenue")
def forecast_revenue(
    user: dict = Depends(require_role(["admin", "analyst", "auditor"]))
):
    """
    Forecast next week's revenue using individual transaction data.
    
    Process:
    1. Fetch all individual transactions for user
    2. Aggregate by week
    3. Create lag features (last 3 weeks predict next)
    4. Train linear regression model
    5. Predict next week
    """
    
    # ── 1. Fetch user ──────────────────────────────────────────
    current_user = users_collection.find_one({"username": user["sub"]})
    if not current_user:
        raise HTTPException(status_code=404, detail="User not found ❌")
 
    user_id = str(current_user["_id"])
 
    # ── 2. Fetch ALL individual transactions (not forecast results) ─
    # ✅ KEY CHANGE: Include ALL transactions with revenue
    data = list(
        financial_collection.find({
            "user_id": user_id,
            "revenue": {"$gt": 0},
            "$or": [
                {"type": {"$exists": False}},
                {"type": {"$ne": "forecast_result"}}
            ]
        }).sort("created_at", 1)
    )
 
    if len(data) < 3:
        return {
            "next_week_prediction": 0,
            "next_month_prediction": 0,
            "model_accuracy_r2": 0,
            "weeks_used_for_training": [],
            "message": f"Need at least 3 transactions for forecasting (found {len(data)})"
        }
 
    # ── 3. Convert to DataFrame for time-series processing ──────
    try:
        df = pd.DataFrame([
            {
                "date": record.get("created_at"),
                "revenue": float(record.get("revenue", 0))
            }
            for record in data
            if record.get("created_at") is not None
        ])
 
        if df.empty or len(df) < 3:
            return {
                "next_week_prediction": 0,
                "next_month_prediction": 0,
                "model_accuracy_r2": 0,
                "weeks_used_for_training": [],
                "message": "Invalid transaction dates"
            }
 
        # Sort by date to ensure chronological order
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
 
        print(f"Processing {len(df)} transactions from {df['date'].min()} to {df['date'].max()}")
 
    except Exception as e:
        print(f"Error converting to DataFrame: {e}")
        return {
            "next_week_prediction": 0,
            "next_month_prediction": 0,
            "model_accuracy_r2": 0,
            "weeks_used_for_training": [],
            "message": f"Error processing dates: {str(e)}"
        }
 
    # ── 4. Aggregate by WEEK (not month) ───────────────────────
    # ✅ KEY CHANGE: Weekly aggregation for better predictions
    df["week"] = df["date"].dt.to_period("W")
    weekly_revenue = df.groupby("week")["revenue"].sum().values.astype(float)
 
    print(f"Aggregated into {len(weekly_revenue)} weeks")
 
    if len(weekly_revenue) < 3:
        return {
            "next_week_prediction": 0,
            "next_month_prediction": 0,
            "model_accuracy_r2": 0,
            "weeks_used_for_training": weekly_revenue.tolist(),
            "message": f"Need at least 3 weeks of data (found {len(weekly_revenue)})"
        }
 
    # ── 5. Create lag features ──────────────────────────────────
    # ✅ KEY CHANGE: Uses last 3 weeks to predict next week
    # Example: [week1, week2, week3] → predict week4
    def create_lag_features(series: np.ndarray, n_lags: int):
        """
        Create lag features for time series prediction.
        
        Args:
            series: Array of weekly revenues
            n_lags: Number of previous weeks to use (default 3)
            
        Returns:
            X: Feature matrix [[week1, week2, week3], [week2, week3, week4], ...]
            y: Target values [week4, week5, week6, ...]
        """
        X, y = [], []
        for i in range(n_lags, len(series)):
            X.append(series[i - n_lags : i])   # Last N weeks as features
            y.append(series[i])                  # Next week as target
        return np.array(X), np.array(y)
 
    # ✅ KEY CHANGE: Dynamic n_lags based on available weeks
    n_lags = min(3, len(weekly_revenue) - 1)   # Use 1-3 previous weeks
    
    print(f"Creating lag features with n_lags={n_lags}")
 
    X, y = create_lag_features(weekly_revenue, n_lags)
 
    if len(X) < 2:
        # Not enough samples for train/test — train on all data
        print("Warning: Less than 2 samples, training on all data")
        model = LinearRegression()
        model.fit(X, y)
        accuracy = float(max(0.0, model.score(X, y)))
        train_samples = len(X)
        test_samples = 0
 
    else:
        # ── 6. Time-series split (80/20, chronological) ─────────
        # ✅ KEY CHANGE: NO shuffling - time order matters!
        split_idx = max(1, int(len(X) * 0.8))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
 
        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
 
        model = LinearRegression()
        model.fit(X_train, y_train)
 
        # ── 7. Calculate real R² accuracy ───────────────────────
        # ✅ KEY CHANGE: Actually compute R², don't hardcode 1.0!
        if len(X_test) > 0:
            y_pred = model.predict(X_test)
            try:
                raw_r2 = r2_score(y_test, y_pred)
                accuracy = float(max(0.0, min(1.0, raw_r2)))  # Clip to [0, 1]
            except Exception as e:
                print(f"Warning: Could not compute R² score: {e}")
                accuracy = float(max(0.0, model.score(X_train, y_train)))
        else:
            accuracy = float(max(0.0, model.score(X_train, y_train)))
 
        train_samples = len(X_train)
        test_samples = len(X_test)
 
    # ── 8. Predict NEXT WEEK ────────────────────────────────────
    # ✅ KEY CHANGE: Use last N weeks to predict next week
    last_window = weekly_revenue[-n_lags:].reshape(1, -1)
    week_prediction = float(model.predict(last_window)[0])
    week_prediction = max(0.0, week_prediction)  # No negative revenue
 
    # ✅ BONUS: Also estimate next month (4 weeks ahead)
    # For months: sum the predicted weeks or use a multiplier
    month_prediction = week_prediction * 4.0  # 4 weeks per month (estimate)
 
    print(f"Prediction: Next week = ₹{week_prediction:,.2f}, Next month ≈ ₹{month_prediction:,.2f}")
 
    # ── 9. Save model & results ─────────────────────────────────
    try:
        joblib.dump(model, "revenue_model.pkl")
    except Exception as e:
        print(f"Warning: Could not save model: {e}")
 
    # Save forecast to database
    try:
        financial_collection.update_one(
            {"user_id": user_id, "type": "forecast_result"},
            {
                "$set": {
                    "prediction_week": round(week_prediction, 2),
                    "prediction_month": round(month_prediction, 2),
                    "accuracy": round(accuracy, 4),
                    "n_lags": n_lags,
                    "weeks_used": len(weekly_revenue),
                    "train_samples": train_samples,
                    "test_samples": test_samples,
                    "created_at": datetime.utcnow()
                }
            },
            upsert=True
        )
    except Exception as e:
        print(f"Warning: Could not save forecast to DB: {e}")
 
    # ── 10. Return comprehensive response ────────────────────────
    return {
        "message": "Revenue forecast completed ✅",
        "next_week_prediction": round(week_prediction, 2),
        "next_month_prediction": round(month_prediction, 2),
        "model_accuracy_r2": round(accuracy, 4),
        "weeks_used_for_training": weekly_revenue.tolist(),
        "lags_used": n_lags,
        "training_samples": train_samples,
        "testing_samples": test_samples,
        "total_transactions": len(data),
        "date_range": {
            "start": df["date"].min().isoformat() if not df.empty else None,
            "end": df["date"].max().isoformat() if not df.empty else None
        }
    }
 
 
# ============================================================================
# ✅ ALTERNATIVE: FORECAST WITH XGBOOST (For more complex patterns)
# ============================================================================
 
@app.get("/forecast-revenue-xgboost")
def forecast_revenue_xgboost(
    user: dict = Depends(require_role(["admin", "analyst", "auditor"]))
):
    """
    Advanced revenue forecasting using XGBoost regressor.
    Better for non-linear patterns and seasonal trends.
    """
    
    current_user = users_collection.find_one({"username": user["sub"]})
    if not current_user:
        raise HTTPException(status_code=404, detail="User not found ❌")
 
    user_id = str(current_user["_id"])
 
    # Fetch individual transactions
    data = list(
        financial_collection.find({
            "user_id": user_id,
            "revenue": {"$gt": 0},
            "$or": [
                {"type": {"$exists": False}},
                {"type": {"$ne": "forecast_result"}}
            ]
        }).sort("created_at", 1)
    )
 
    if len(data) < 5:
        return {"message": "Need at least 5 transactions", "prediction": 0}
 
    # Create time series
    df = pd.DataFrame([
        {
            "date": record.get("created_at"),
            "revenue": float(record.get("revenue", 0))
        }
        for record in data
    ])
 
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
 
    # Weekly aggregation
    df["week"] = df["date"].dt.to_period("W")
    weekly_revenue = df.groupby("week")["revenue"].sum().values.astype(float)
 
    if len(weekly_revenue) < 5:
        return {"message": "Need at least 5 weeks of data", "prediction": 0}
 
    # Create advanced features
    def create_advanced_features(series, n_lags=3):
        X, y = [], []
        for i in range(n_lags, len(series)):
            # Lag features
            lags = list(series[i-n_lags:i])
            
            # Rolling statistics
            window = series[max(0, i-7):i]  # Last 7 weeks
            rolling_mean = float(np.mean(window)) if len(window) > 0 else 0
            rolling_std = float(np.std(window)) if len(window) > 1 else 0
            
            # Trend
            if i >= 2:
                trend = (series[i-1] - series[i-2]) / (series[i-2] + 0.01)
            else:
                trend = 0
            
            # Combine features
            features = lags + [rolling_mean, rolling_std, trend]
            X.append(features)
            y.append(series[i])
        
        return np.array(X), np.array(y)
 
    X, y = create_advanced_features(weekly_revenue, n_lags=3)
 
    if len(X) < 3:
        return {"message": "Insufficient data for advanced forecasting", "prediction": 0}
 
    # Train/test split
    split_idx = max(1, int(len(X) * 0.8))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
 
    # Train XGBoost model
    from xgboost import XGBRegressor
    
    model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    
    model.fit(X_train, y_train)
 
    # Evaluate
    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        accuracy = float(r2_score(y_test, y_pred))
    else:
        accuracy = float(model.score(X_train, y_train))
 
    # Predict next week
    last_features = X[-1].reshape(1, -1)  # Last observed features
    prediction = float(model.predict(last_features)[0])
    prediction = max(0.0, prediction)
 
    return {
        "message": "XGBoost forecast completed ✅",
        "next_week_prediction": round(prediction, 2),
        "next_month_prediction": round(prediction * 4, 2),
        "model_accuracy_r2": round(accuracy, 4),
        "model_type": "XGBoost (advanced)",
        "weeks_used": len(weekly_revenue),
        "total_transactions": len(data)
    }
 
 
# ============================================================================
# ✅ UTILITY: GET FORECAST STATS
# ============================================================================
 
@app.get("/forecast-stats")
def get_forecast_stats(
    user: dict = Depends(require_role(["admin", "analyst", "auditor"]))
):
    """Get latest forecast statistics for dashboard"""
    
    current_user = users_collection.find_one({"username": user["sub"]})
    if not current_user:
        raise HTTPException(status_code=404, detail="User not found ❌")
 
    user_id = str(current_user["_id"])
 
    # Fetch latest forecast
    forecast = financial_collection.find_one(
        {"user_id": user_id, "type": "forecast_result"},
        sort=[("created_at", -1)]
    )
 
    if not forecast:
        return {
            "has_forecast": False,
            "next_week_prediction": 0,
            "next_month_prediction": 0,
            "accuracy": 0
        }
 
    return {
        "has_forecast": True,
        "next_week_prediction": forecast.get("prediction_week", 0),
        "next_month_prediction": forecast.get("prediction_month", 0),
        "accuracy": forecast.get("accuracy", 0),
        "weeks_used": forecast.get("weeks_used", 0),
        "created_at": forecast.get("created_at").isoformat() if forecast.get("created_at") else None
    }
 
 # ---------------- XGBOOST RISK CLASSIFICATION ----------------

@app.get("/classify-risk-xgb")
def classify_risk_xgb(
    user: dict = Depends(require_role(["admin", "analyst", "auditor"]))
):
    try:
        # 🔹 Get logged-in user
        current_user = users_collection.find_one({"username": user["sub"]})

        if not current_user:
            raise HTTPException(status_code=404, detail="User not found ❌")

        user_id = str(current_user["_id"])

        # 🔹 Get financial data
        data = list(financial_collection.find({
            "user_id": user_id
        }))

        if not data:
            return {
                "message": "No financial data uploaded yet",
                "total_records": 0,
                "results": []
            }

        # 🔹 Prepare valid data
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

        # 🔥 IMPORTANT FIX: allow small dataset
        if len(valid_data) < 2:
            return {
                "message": "Not enough data, assigning default risk",
                "total_records": len(valid_data),
                "results": []
            }

        # -------- Feature Matrix --------
        revenues = np.array(
            [d["revenue"] for d in valid_data]
        ).reshape(-1, 1)

        # -------- Rule-based fallback (ALWAYS WORKS) --------
        labels = []
        for r in revenues:
            value = r[0]

            if value < 1000:
                labels.append("Low")
            elif value < 5000:
                labels.append("Medium")
            else:
                labels.append("High")

        # 🔥 If dataset small → skip ML and directly assign
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

        # -------- ML PART --------
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

        # -------- Evaluate --------
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # -------- Predict --------
        predictions = model.predict(revenues)

        results = []

        for i, record in enumerate(valid_data):

            risk_label = encoder.inverse_transform([predictions[i]])[0]

            # 🔹 SAVE TO DB (CRITICAL)
            financial_collection.update_one(
                {"_id": record["_id"]},
                {"$set": {"risk_level": risk_label}}
            )

            results.append({
                "id": str(record["_id"]),
                "revenue": record["revenue"],
                "risk": risk_label
            })

        # 🔹 Save model (optional)
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

# ADD BLOCK API

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

# VIEW BLOCKCHAIN

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

# VERIFY BLOCKCHAIN

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

# KPI API

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

# REVENUE FORECAST (GRAPH DATA)

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

# CHART DATA

@app.get("/chart-data")
def chart_data(
    user: dict = Depends(require_role(["admin","analyst","auditor"]))
):

    # 🔹 Get logged-in user
    current_user = users_collection.find_one({"username": user["sub"]})

    if not current_user:
        raise HTTPException(status_code=404, detail="User not found ❌")

    # 🔹 Get financial data for that user
    data = list(
        financial_collection.find({
            "user_id": str(current_user["_id"])
        })
    )

    if not data:
        return []

    today = datetime.today()

    months = [
        (today - relativedelta(months=i)).strftime("%b")
        for i in range(5, -1, -1)
    ]

    revenues = []
    expenses = []

    # 🔹 Safe data extraction
    for r in data:
        try:
            revenues.append(float(r.get("revenue", 0)))
        except:
            revenues.append(0)

        try:
            expenses.append(float(r.get("expense", 0)))
        except:
            expenses.append(0)

    chunk_size = max(1, len(revenues) // 6)

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

@app.get("/dashboard-data")
def get_dashboard_data(
    user: dict = Depends(require_role(["admin","analyst","auditor"]))
):

    try:
        # 🔹 Get user
        current_user = users_collection.find_one({"username": user["sub"]})

        if not current_user:
            raise HTTPException(status_code=404, detail="User not found ❌")

        user_id = str(current_user["_id"])

        # 🔥 IMPORTANT: FETCH DATA ONLY ONCE
        data = list(financial_collection.find({
            "user_id": user_id
        }))

        # 🔹 Empty case
        if not data:
            return {
                "kpis": {
                    "total_revenue": 0,
                    "total_expense": 0,
                    "net_profit": 0
                },
                "forecast": [],
                "chart": [],
                "prediction": {
                    "next_month_prediction": 0,
                    "model_accuracy_r2": 0
                },
                "anomaly": {
                    "high": 0,
                    "medium": 0,
                    "low": 0
                },
                "blockchain": {
                    "status": "Unknown"
                }
            }

        # =========================
        # 🔥 KPI CALCULATION
        # =========================

        total_revenue = 0
        total_expense = 0

        for r in data:
            try:
                total_revenue += float(r.get("revenue", 0))
            except:
                pass

            try:
                total_expense += float(r.get("expense", 0))
            except:
                pass

        kpis = {
            "total_revenue": total_revenue,
            "total_expense": total_expense,
            "net_profit": total_revenue - total_expense
        }

        # =========================
        # 🔥 RISK COUNT
        # =========================

        high = medium = low = 0

        for r in data:
            risk = r.get("risk_level")

            if risk == "High":
                high += 1
            elif risk == "Medium":
                medium += 1
            elif risk == "Low":
                low += 1

        anomaly = {
            "high": high,
            "medium": medium,
            "low": low
        }

        # =========================
        
        # 🔥 PREDICTION
        # =========================

        prediction_data = financial_collection.find_one({
            "user_id": user_id,
            "type": "forecast_result"
        })

        prediction = {
            "next_month_prediction": 0,
            "model_accuracy_r2": 0
        }

        if prediction_data:
            prediction = {
                "next_month_prediction": float(prediction_data.get("prediction", 0)),
                "model_accuracy_r2": float(prediction_data.get("accuracy", 0))
            }

        # =========================
        # 🔥 BLOCKCHAIN
        # =========================

        try:
            bc = verify_integrity(user)
            blockchain_status = bc.get("status", "Unknown")
        except:
            blockchain_status = "Unknown"

        # =========================
        # 🔥 FINAL RESPONSE (ALL AT ONCE)
        # =========================

        return {
            "kpis": kpis,
            "forecast": revenue_forecast(user),
            "chart": chart_data(user),
            "prediction": prediction,
            "anomaly": anomaly,
            "blockchain": {
                "status": blockchain_status
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Dashboard error: {str(e)}"
        )
        
