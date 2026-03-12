from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://anonymousmalware38_db_user:Malware%40127%23137@cluster0.lagn1ln.mongodb.net/?retryWrites=true&w=majority"

client = MongoClient(uri, server_api=ServerApi('1'))

# Database
db = client["finpulse_db"]

# Collections
users_collection = db["users"]
financial_collection = db["financial_data"]
ai_collection = db["ai_insights"]
fraud_collection = db["fraud_alerts"]
reports_collection = db["reports"]

# 🔥 ADD THIS
blockchain_collection = db["blockchain"]