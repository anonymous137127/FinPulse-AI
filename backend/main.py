from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.auth_routes import router as auth_router
from app.routes.data_routes import router as data_router
from app.routes.ml_routes import router as ml_router
from app.routes.dashboard_routes import router as dashboard_router
from app.routes.blockchain_routes import router as blockchain_router

app = FastAPI(title="FinPulse AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "FinPulse Backend Running 🚀"}

app.include_router(auth_router)
app.include_router(data_router)
app.include_router(ml_router)
app.include_router(dashboard_router)
app.include_router(blockchain_router)