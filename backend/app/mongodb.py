import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")

client = MongoClient(MONGO_URI)

db = client[DATABASE_NAME]

# Collections
users_collection = db["users"]
financial_collection = db["financial_data"]
ai_collection = db["ai_insights"]
fraud_collection = db["fraud_alerts"]
reports_collection = db["reports"]
blockchain_collection = db["blockchain"]