import os
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

load_dotenv()

# Get values from .env
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")

# MongoDB Client
client = MongoClient(MONGO_URI, server_api=ServerApi('1'))

# Database
db = client[DATABASE_NAME]

# Collections
users_collection = db["users"]
financial_collection = db["financial_data"]
ai_collection = db["ai_insights"]
fraud_collection = db["fraud_alerts"]
reports_collection = db["reports"]
blockchain_collection = db["blockchain"]