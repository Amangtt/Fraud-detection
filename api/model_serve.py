from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load your model and scaler
model = joblib.load('./model/fraud_rf_model.pkl')
scaler = joblib.load('./model/scaler.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return "Fraud Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if isinstance(data, dict):
        data = {k: [v] for k, v in data.items()}

    df = pd.DataFrame(data)

    # Define expected features
    numeric_features = ['purchase_value', 'age', 'hour_of_day', 'day_of_week',
                        'purchase_delay', 'user_transaction_frequency',
                        'device_transaction_frequency', 'user_transaction_velocity']
    categorical_features = ['sex_M','browser_FireFox','browser_IE','browser_Opera', 'browser_Safari','source_Direct', 'source_SEO']  

    # Ensure all features are present
    expected_features = numeric_features + categorical_features
    # Check for missing features
    for feature in expected_features:
        if feature not in df.columns:
            return jsonify({"error": f"Missing feature: {feature}"}), 400

    # Scale only numeric features
    df[numeric_features] = scaler.transform(df[numeric_features])

    # Make predictions
    prediction = model.predict(df)
    probability = model.predict_proba(df)[:, 1]

    return jsonify({"prediction": int(prediction[0]), "fraud_probability": float(probability[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)