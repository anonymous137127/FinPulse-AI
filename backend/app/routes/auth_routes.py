from fastapi import APIRouter, HTTPException
from passlib.context import CryptContext
from datetime import datetime

from app.mongodb import users_collection
from app.auth import create_access_token

router = APIRouter()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@router.post("/register")
def register(username: str, password: str, role: str):

    if users_collection.find_one({"username": username}):
        raise HTTPException(status_code=400, detail="Username exists ❌")

    hashed_password = pwd_context.hash(password)

    users_collection.insert_one({
        "username": username,
        "password": hashed_password,
        "role": role,
        "created_at": datetime.utcnow()
    })

    return {"message": "User registered successfully ✅"}


@router.post("/login")
def login(username: str, password: str):

    user = users_collection.find_one({"username": username})

    if not user:
        raise HTTPException(status_code=401, detail="Invalid login ❌")

    if not pwd_context.verify(password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid login ❌")

    token = create_access_token({
        "sub": user["username"],
        "role": user["role"]
    })

    return {
        "access_token": token,
        "token_type": "bearer"
    }