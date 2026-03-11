from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.orm import declarative_base
from app.database import engine

Base = declarative_base()

# ---------------- USER TABLE ----------------
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True)
    password = Column(String(200))
    role = Column(String(50))


# ---------------- FINANCIAL DATA TABLE ----------------
class FinancialData(Base):
    __tablename__ = "financial_data"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    revenue = Column(String(100))
    expense = Column(String(100))

    # Risk column
    risk_level = Column(String(50), default="Normal")


# ---------------- REVENUE PREDICTION TABLE ----------------
class RevenuePrediction(Base):
    __tablename__ = "revenue_predictions"

    id = Column(Integer, primary_key=True, index=True)
    predicted_value = Column(String(100))
    created_at = Column(String(100))


# ---------------- BLOCKCHAIN TABLE ----------------
class BlockchainBlock(Base):
    __tablename__ = "blockchain"

    id = Column(Integer, primary_key=True, index=True)
    block_index = Column(Integer)
    timestamp = Column(String(100))
    data = Column(Text)
    previous_hash = Column(String(255))
    current_hash = Column(String(255))


# 🔥 Create ALL tables AFTER all models defined
Base.metadata.create_all(bind=engine)