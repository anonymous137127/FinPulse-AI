from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import pandas as pd
import hashlib
import json

# Optional imports from your project
from app.database import SessionLocal
from app import models
from app.mongodb import db as mongo_db

from sqlalchemy.orm import Session

# ---------------- SECURITY ----------------

SECRET_KEY = "my_super_secret_key_123"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# ---------------- FASTAPI ----------------

app = FastAPI(title="FinPulse API 🚀")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- DATABASE ----------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------- AUTH ----------------

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})

    token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return token


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):

    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload

    except:
        raise HTTPException(status_code=401, detail="Invalid token")


def require_role(roles: list):

    def role_checker(user: dict = Depends(verify_token)):

        if user.get("role") not in roles:
            raise HTTPException(status_code=403, detail="Access denied")

        return user

    return role_checker

# ---------------- BASIC ROUTES ----------------

@app.get("/")
def home():
    return {"message": "FinPulse Backend Running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------- MONGO TEST ----------------

@app.get("/mongo-test")
def mongo_test():

    try:

        data = {"message": "MongoDB working"}

        mongo_db.test.insert_one(data)

        return {"status": "MongoDB Insert Success"}

    except Exception as e:

        return {"error": str(e)}

# ---------------- REGISTER ----------------

@app.post("/register")
def register(
    username: str,
    password: str,
    role: str,
    db: Session = Depends(get_db)
):

    username = username.strip()
    password = password.strip()

    if role not in ["admin", "analyst", "auditor"]:
        raise HTTPException(status_code=400, detail="Invalid role")

    existing_user = db.query(models.User).filter(
        models.User.username == username
    ).first()

    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")

    hashed_password = pwd_context.hash(password)

    new_user = models.User(
        username=username,
        password=hashed_password,
        role=role
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {
        "message": "User registered successfully",
        "username": username,
        "role": role
    }

# ---------------- LOGIN ----------------

@app.post("/login")
def login(
    username: str,
    password: str,
    db: Session = Depends(get_db)
):

    user = db.query(models.User).filter(
        models.User.username == username
    ).first()

    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    if not pwd_context.verify(password, user.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_access_token({
        "sub": user.username,
        "role": user.role
    })

    return {
        "access_token": token,
        "token_type": "bearer"
    }

# ---------------- CSV UPLOAD ----------------

@app.post("/upload-csv")
def upload_csv(
    file: UploadFile = File(...),
    user: dict = Depends(require_role(["admin", "analyst", "auditor"])),
    db: Session = Depends(get_db)
):

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")

    try:

        df = pd.read_csv(file.file)

    except Exception:

        raise HTTPException(status_code=400, detail="Invalid CSV file")

    rows = []

    for _, row in df.iterrows():

        try:

            rows.append(
                models.FinancialData(
                    revenue=float(row[0]),
                    expense=float(row[1])
                )
            )

        except:
            continue

    db.bulk_save_objects(rows)
    db.commit()

    return {
        "message": "CSV uploaded successfully",
        "rows_inserted": len(rows)
    }

# ---------------- KPI ----------------

@app.get("/kpis")
def get_kpis(
    user: dict = Depends(require_role(["admin", "analyst", "auditor"])),
    db: Session = Depends(get_db)
):

    data = db.query(models.FinancialData).all()

    total_revenue = sum(float(r.revenue) for r in data)
    total_expense = sum(float(r.expense) for r in data)

    return {
        "total_revenue": total_revenue,
        "total_expense": total_expense,
        "net_profit": total_revenue - total_expense
    }

# ---------------- HASH DATA ----------------

@app.get("/hash-financial-data")
def hash_financial_data(
    user: dict = Depends(require_role(["admin","analyst","auditor"])),
    db: Session = Depends(get_db)
):

    data = db.query(models.FinancialData).all()

    dataset = []

    for record in data:

        dataset.append({
            "id": record.id,
            "revenue": record.revenue,
            "expense": record.expense
        })

    dataset_string = json.dumps(dataset, sort_keys=True)

    dataset_hash = hashlib.sha256(dataset_string.encode()).hexdigest()

    return {
        "sha256_hash": dataset_hash,
        "total_records": len(dataset)
    }