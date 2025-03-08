from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

class EconomicFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, add_interaction=True):
        self.add_interaction = add_interaction
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=[
            'unemployment_rate', 'personal_consumption',
            'govt_expenditure', 'm1_money_supply',
            'm2_money_supply', 'federal_debt'
        ])
        X_scaled = self.scaler.transform(X_df)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X_df.columns)
        
        result = X_scaled_df.copy()
        
        if self.add_interaction:
            result['consumption_ratio'] = X_scaled_df['personal_consumption'] / (
                X_scaled_df['personal_consumption'] + X_scaled_df['govt_expenditure']
            )
            result['m2_m1_ratio'] = X_scaled_df['m2_money_supply'] / X_scaled_df['m1_money_supply'].replace(0, 0.001)
            result['debt_spending_ratio'] = X_scaled_df['federal_debt'] / (
                X_scaled_df['personal_consumption'] + X_scaled_df['govt_expenditure']
            ).replace(0, 0.001)
            result['unemployment_consumption'] = X_scaled_df['unemployment_rate'] * X_scaled_df['personal_consumption']
            result['log_consumption'] = np.log1p(np.abs(X_scaled_df['personal_consumption']))
            result['log_govt_exp'] = np.log1p(np.abs(X_scaled_df['govt_expenditure']))
        
        return result.values

MODEL = None
REQUIRED_FEATURES = [
    'unemployment_rate',
    'personal_consumption',
    'govt_expenditure',
    'm1_money_supply',
    'm2_money_supply',
    'federal_debt'
]

def load_model():
    global MODEL
    if MODEL is None:
        MODEL = joblib.load("models/elastic_net_model.pkl")
    return True

@app.route('/api/predict', methods=['POST'])
def predict_gdp():
    if not load_model():
        return jsonify({
            "error": "Model Error",
            "message": "Could not load prediction model"
        }), 500

    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        input_values = {}
        for feature in REQUIRED_FEATURES:
            value = data.get(feature)
            if value is None:
                return jsonify({
                    "error": "Missing field",
                    "message": f"{feature} is required"
                }), 400
            
            try:
                num_value = float(value)
                if feature == 'unemployment_rate' and not (0 <= num_value <= 100):
                    return jsonify({
                        "error": "Invalid value",
                        "message": "Unemployment rate must be between 0-100"
                    }), 400
                input_values[feature] = num_value
            except ValueError:
                return jsonify({
                    "error": "Invalid value",
                    "message": f"{feature} must be a number"
                }), 400

        input_array = np.array([[input_values[feat] for feat in REQUIRED_FEATURES]])
        prediction = (MODEL.predict(input_array)[0])*5
        
        return jsonify({
            "gdp_prediction": float(prediction),
            "currency": "₹ Crores",
            "model_used": "elastic_net",
            "input_values": input_values
        })
        
    except Exception as e:
        return jsonify({
            "error": "Prediction Failed",
            "message": str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "running",
        "model_loaded": MODEL is not None
    })

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    load_model()
    app.run(host='0.0.0.0', port=os.getenv('PORT', 5000))