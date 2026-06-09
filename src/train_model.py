import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_preprocessing import load_and_clean_data
from feature_engineering import create_features

# Load data
df = load_and_clean_data('../data/household_power_consumption.csv')

# Clean columns
df.columns = df.columns.str.strip()

# Feature engineering
df = create_features(df)

# Required columns
required_cols = ['hour', 'day', 'month', 'Global_active_power']
for col in required_cols:
    if col not in df.columns:
        raise Exception(f"Column missing: {col}")

# Features & target
X = df[['hour', 'day', 'month']]
y = df['Global_active_power'].astype(float)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# OPTIMIZED MODEL (KEY CHANGE)
model = RandomForestRegressor(
    n_estimators=30,        # ↓ from 100 → reduces size massively
    max_depth=10,           # control tree size
    min_samples_split=10,   # avoid over-complex trees
    random_state=42,
    n_jobs=-1
)

# Optional pipeline (clean & efficient)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', model)
])

# Train
pipeline.fit(X_train, y_train)

# COMPRESS MODEL (VERY IMPORTANT)
joblib.dump(pipeline, '../model/model.pkl', compress=3)

print("Optimized Model Saved")