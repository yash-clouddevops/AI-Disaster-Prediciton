"""
Create all prediction models: Flood, Earthquake, Hurricane.
Run once: python create_all_models.py
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)

# ----- Flood (same as before) -----
print("Training Flood model...")
np.random.seed(42)
n = 2000
df = pd.DataFrame({
    'Latitude': np.random.uniform(8, 37, n),
    'Longitude': np.random.uniform(68, 97, n),
    'Rainfall (mm)': np.random.uniform(0, 300, n),
    'Temperature (°C)': np.random.uniform(15, 45, n),
    'Humidity (%)': np.random.uniform(20, 100, n),
    'River Discharge (m³/s)': np.random.uniform(0, 5000, n),
    'Water Level (m)': np.random.uniform(0, 10, n),
    'Elevation (m)': np.random.uniform(1, 9000, n),
    'Land Cover': np.random.choice(['Agricultural', 'Desert', 'Forest', 'Urban', 'Water Body'], n),
    'Soil Type': np.random.choice(['Clay', 'Loam', 'Peat', 'Sandy', 'Silt'], n),
    'Population Density': np.random.uniform(2, 10000, n),
    'Infrastructure': np.random.choice([0, 1], n),
    'Historical Floods': np.random.choice([0, 1], n),
})
flood_risk = (df['Rainfall (mm)'] > 150) & (df['Water Level (m)'] > 5)
df['Flood Occurred'] = (flood_risk.astype(int) + np.random.choice([0, 1], n, p=[0.3, 0.7])).clip(0, 1)
X = df.drop('Flood Occurred', axis=1)
y = df['Flood Occurred']
X_enc = pd.get_dummies(X, drop_first=True)
joblib.dump(RandomForestClassifier(n_estimators=100, random_state=42).fit(X_enc, y), "random_forest_model.pkl")
joblib.dump(X_enc.columns.tolist(), "model_features.pkl")
print("  -> random_forest_model.pkl, model_features.pkl")

# ----- Earthquake -----
print("Training Earthquake model...")
eq_path = os.path.join(BASE, "earthquake (1).csv")
if os.path.exists(eq_path):
    eq = pd.read_csv(eq_path)
    eq = eq[['Latitude', 'Longitude', 'Magnitude']].dropna()
    X_eq = eq[['Latitude', 'Longitude']]
    y_eq = eq['Magnitude']
    joblib.dump(RandomForestRegressor(n_estimators=100, random_state=42).fit(X_eq, y_eq), "earthquake_model.pkl")
    print("  -> earthquake_model.pkl")
else:
    # Fallback synthetic
    X_eq = pd.DataFrame({'Latitude': np.random.uniform(-60, 60, 1000), 'Longitude': np.random.uniform(-180, 180, 1000)})
    y_eq = 4 + np.random.randn(1000) * 1.5
    joblib.dump(RandomForestRegressor(n_estimators=100, random_state=42).fit(X_eq, y_eq), "earthquake_model.pkl")
    print("  -> earthquake_model.pkl (synthetic)")

# ----- Hurricane -----
print("Training Hurricane model...")
hur_path = os.path.join(BASE, "atlantic.csv")
wind_cols = ['Moderate Wind NE', 'Moderate Wind SE', 'Moderate Wind SW', 'Moderate Wind NW']
target_col = 'Maximum Wind'
if os.path.exists(hur_path):
    hur = pd.read_csv(hur_path)
    # Parse Latitude "28.0N" -> 28.0, "94.8W" -> -94.8
    def parse_lat(s):
        s = str(s).strip().upper()
        if not s or s == 'NAN': return np.nan
        try:
            v = float(''.join(c for c in s if c in '0123456789.-' or c == '.'))
            if 'S' in s: v = -v
            return v
        except: return np.nan
    def parse_lon(s):
        s = str(s).strip().upper()
        if not s or s == 'NAN': return np.nan
        try:
            v = float(''.join(c for c in s if c in '0123456789.-' or c == '.'))
            if 'W' in s: v = -v
            return v
        except: return np.nan
    hur['Latitude'] = hur['Latitude'].apply(parse_lat)
    hur['Longitude'] = hur['Longitude'].apply(parse_lon)
    hur['Date'] = hur['Date'].astype(str)
    hur['Year'] = pd.to_numeric(hur['Date'].str[:4], errors='coerce').fillna(2000).astype(int)
    hur['Month'] = pd.to_numeric(hur['Date'].str[4:6], errors='coerce').fillna(6).astype(int)
    hur['Day'] = pd.to_numeric(hur['Date'].str[6:8], errors='coerce').fillna(15).astype(int)
    feat_cols = ['Latitude', 'Longitude'] + wind_cols + ['Year', 'Month', 'Day']
    for c in feat_cols:
        if c not in hur.columns:
            hur[c] = 0
    hur = hur[feat_cols + [target_col]].replace(-999, np.nan)
    hur = hur.fillna(hur.median(numeric_only=True))
    hur = hur.dropna()
    if len(hur) < 100:
        raise ValueError("Not enough hurricane data after cleaning")
    X_hur = hur[feat_cols]
    y_hur = hur[target_col]
    joblib.dump(RandomForestRegressor(n_estimators=100, random_state=42).fit(X_hur, y_hur), "hurricane_model.pkl")
    joblib.dump(feat_cols, "hurricane_features.pkl")
    print("  -> hurricane_model.pkl, hurricane_features.pkl")
else:
    X_hur = pd.DataFrame({
        'Latitude': np.random.uniform(10, 35, 500),
        'Longitude': np.random.uniform(-100, -70, 500),
        'Moderate Wind NE': np.random.uniform(0, 100, 500),
        'Moderate Wind SE': np.random.uniform(0, 100, 500),
        'Moderate Wind SW': np.random.uniform(0, 100, 500),
        'Moderate Wind NW': np.random.uniform(0, 100, 500),
        'Year': np.random.randint(2000, 2024, 500),
        'Month': np.random.randint(1, 13, 500),
        'Day': np.random.randint(1, 29, 500),
    })
    y_hur = 30 + np.random.randn(500) * 25
    joblib.dump(RandomForestRegressor(n_estimators=100, random_state=42).fit(X_hur, y_hur), "hurricane_model.pkl")
    joblib.dump(X_hur.columns.tolist(), "hurricane_features.pkl")
    print("  -> hurricane_model.pkl, hurricane_features.pkl (synthetic)")

print("Done. All models created.")
