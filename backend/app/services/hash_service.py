import hashlib
import json
from datetime import datetime

def generate_ml_hash(records):

    results = []

    high = 0
    medium = 0
    low = 0

    for record in records:

        risk = record.get("risk_level", "Unknown")

        if risk == "High":
            high += 1
        elif risk == "Medium":
            medium += 1
        elif risk == "Low":
            low += 1

        results.append({
            "id": str(record["_id"]),
            "risk_level": risk
        })

    results_string = json.dumps(results, sort_keys=True)

    hash_value = hashlib.sha256(results_string.encode()).hexdigest()

    return hash_value, high, medium, low, results