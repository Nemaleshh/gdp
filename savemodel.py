import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os

REQUIRED_FEATURES = [
    'unemployment_rate',
    'personal_consumption',
    'govt_expenditure',
    'm1_money_supply',
    'm2_money_supply',
    'federal_debt'
]

class EconomicFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, add_interaction=True):
        self.add_interaction = add_interaction
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=REQUIRED_FEATURES)
        X_scaled = self.scaler.transform(X_df)
        X_scaled_df = pd.DataFrame(X_scaled, columns=REQUIRED_FEATURES)
        
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

def train_gdp_model(historical_data):
    if not all(feat in historical_data.columns for feat in REQUIRED_FEATURES):
        missing = [feat for feat in REQUIRED_FEATURES if feat not in historical_data.columns]
        raise ValueError(f"Missing required features: {missing}")

    X = historical_data[REQUIRED_FEATURES]
    y = historical_data['gdp']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('transformer', EconomicFeatureTransformer()),
        ('model', ElasticNet(random_state=42))
    ])

    param_grid = {
        'model__alpha': [0.01, 0.1, 1.0],
        'model__l1_ratio': [0.2, 0.5, 0.8],
        'transformer__add_interaction': [True, False]
    }

    print("Training ElasticNet...")
    grid_search = GridSearchCV(
        pipeline, param_grid,
        cv=5, scoring='neg_mean_squared_error', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_estimator = grid_search.best_estimator_
    y_pred = best_estimator.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"  R²: {r2:.4f}")
    print(f"  RMSE: {rmse:,.2f}")
    print(f"  Best params: {grid_search.best_params_}")

    os.makedirs('models', exist_ok=True)
    model_path = "models/elastic_net_model.pkl"
    joblib.dump(best_estimator, model_path)
    
    return best_estimator

def generate_synthetic_data(n_samples=2000):
    np.random.seed(42)
    data = {
        'unemployment_rate': np.random.normal(5, 2, n_samples).clip(0, 30),
        'personal_consumption': np.random.normal(20000, 5000, n_samples).clip(1000),
        'govt_expenditure': np.random.normal(8000, 2000, n_samples).clip(500),
        'm1_money_supply': np.random.normal(4000, 1000, n_samples).clip(100),
        'm2_money_supply': lambda x: x['m1_money_supply'] + np.random.normal(4000, 800, n_samples).clip(0),
        'federal_debt': np.random.normal(20000, 8000, n_samples).clip(1000)
    }
    data['m2_money_supply'] = data['m2_money_supply'](data)
    
    df = pd.DataFrame(data)
    df['gdp'] = (
        df['personal_consumption'] * 0.68 +
        df['govt_expenditure'] * 0.25 +
        (df['m1_money_supply'] * 0.1 + df['m2_money_supply'] * 0.05) -
        df['unemployment_rate'] * 200 +
        df['federal_debt'] * 0.02 +
        np.random.normal(0, df['personal_consumption'] * 0.05)
    )
    
    os.makedirs('models', exist_ok=True)
    df.to_csv("models/synthetic_training_data.csv", index=False)
    return df

def main():
    os.makedirs('models', exist_ok=True)
    
    print("Generating synthetic training data...")
    training_data = generate_synthetic_data()
    
    print("\nTraining GDP prediction model...")
    train_gdp_model(training_data)
    
    print("\nTraining completed successfully!")
    print(f"Model saved to: models/elastic_net_model.pkl")

if __name__ == "__main__":
    main()