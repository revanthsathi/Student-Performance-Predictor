import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv("student_data.csv")

features = [
    "study_hours",
    "attendance",
    "previous_marks",
    "sleep_hours",
    "assignments_completed",
    "class_participation",
    "screen_time",
    "extracurricular_hours",
    "stress_level"
]

X = data[features]

y = data["final_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

r2 = r2_score(y_test, predictions)

print("Model trained successfully!")

print("\nModel Performance")

print("Mean Absolute Error:", round(mae, 2))

print("R2 Score:", round(r2, 2))

joblib.dump(model, "student_model.pkl")

joblib.dump(features, "features.pkl")

print("\nModel saved successfully!")