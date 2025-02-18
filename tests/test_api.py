import requests
import json

# API URL (update if running on a different host/port)
API_URL = "http://127.0.0.1:5001/predict"

# Sample input data
data = {
    "purchase_value": 15,
    "age": 53,
    "hour_of_day": 18,
    "day_of_week": 3,
    "purchase_delay": 0.0002777777777777780,
    "user_transaction_frequency": 1,
    "device_transaction_frequency": 12,
    "user_transaction_velocity": 3600.0,
    "sex_M": 1,
    "browser_FireFox": 0,
    "browser_IE": 0,
    "browser_Opera": 0,
    "browser_Safari": 1,
    "source_Direct": 1,
    "source_SEO": 0
}

# Send POST request
response = requests.post(API_URL, json=data)

# Print response
if response.status_code == 200:
    print("Response JSON:", response.json())
else:
    print("Error:", response.status_code, response.text)
