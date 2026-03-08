"""Create model and model_features .pkl files so app.py can run without the original dataset."""
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
n = 2000

# Synthetic data matching the notebook's columns and dtypes
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

# Simple rule so model is not random: high rain + high water level -> more likely flood
flood_risk = (df['Rainfall (mm)'] > 150) & (df['Water Level (m)'] > 5)
df['Flood Occurred'] = (flood_risk.astype(int) + np.random.choice([0, 1], n, p=[0.3, 0.7])).clip(0, 1)

X = df.drop('Flood Occurred', axis=1)
y = df['Flood Occurred']

# Match notebook: get_dummies with drop_first=True
X_encoded = pd.get_dummies(X, drop_first=True)
feature_columns = X_encoded.columns.tolist()

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_encoded, y)

joblib.dump(model, "random_forest_model.pkl")
joblib.dump(feature_columns, "model_features.pkl")
print("Created random_forest_model.pkl and model_features.pkl")
