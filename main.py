from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import matplotlib.pyplot as plt
import io
import base64

# Load the trained model and scaler
model = tf.keras.models.load_model("health_model1.keras")
scaler = joblib.load("scaler.pkl")  # Ensure this file exists

# Initialize FastAPI app
app = FastAPI(title="Health Risk Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Setup
conn = sqlite3.connect("health_data.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS health_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ecg_value REAL,
        glucose_value REAL,
        predicted_risk TEXT,
        probability_score REAL
    )
""")
conn.commit()

# Define request body format
class HealthInput(BaseModel):
    ecg_value: float
    glucose_value: float

# Define health risk labels
RISK_LABELS = ["Normal", "Elevated", "High"]

@app.post("/predict")
def predict_health(data: HealthInput):
    input_data = np.array([[data.ecg_value, data.glucose_value]])
    input_scaled = scaler.transform(input_data)
    probability = model.predict(input_scaled)[0][0]
    risk_level = RISK_LABELS[int(probability > 0.5) + int(probability > 0.8)]
    
    insights = get_health_insights(data.ecg_value, data.glucose_value)
    recommendations = get_recommendations(data.ecg_value, data.glucose_value)
    
    # Store data in the database
    cursor.execute(
        "INSERT INTO health_records (ecg_value, glucose_value, predicted_risk, probability_score) VALUES (?, ?, ?, ?)",
        (data.ecg_value, data.glucose_value, risk_level, round(float(probability), 4))
    )
    conn.commit()
    
    return {
        "message": "Prediction successful",
        "input": data.dict(),
        "predicted_risk": risk_level,
        "probability_score": round(float(probability), 4),
        "insights": insights,
        "recommendations": recommendations
    }

@app.get("/data")
def get_all_data():
    cursor.execute("SELECT * FROM health_records")
    records = cursor.fetchall()
    return {"data": records}

@app.get("/plot")
def plot_data():
    cursor.execute("SELECT ecg_value, glucose_value FROM health_records")
    data = cursor.fetchall()
    if not data:
        return {"message": "No data available to plot."}
    
    ecg_values, glucose_values = zip(*data)
    plt.figure(figsize=(8, 5))
    plt.plot(ecg_values, label="ECG Values", marker="o")
    plt.plot(glucose_values, label="Glucose Values", marker="s")
    plt.xlabel("Entries")
    plt.ylabel("Values")
    plt.legend()
    plt.title("ECG & Glucose Level Trends")
    
    img_io = io.BytesIO()
    plt.savefig(img_io, format="png")
    img_io.seek(0)
    encoded_img = base64.b64encode(img_io.read()).decode()
    
    return {"plot": encoded_img}
    plt.savefig("health_plot.png")

# Health insights function
def get_health_insights(ecg_value, glucose_value):
    insights = {}
    
    # ECG Insights
    if ecg_value < 0.8:
        insights["ECG"] = "ECG value is lower than normal, which may indicate irregular heart rhythms. Please consult a healthcare provider."
    elif ecg_value > 1.0:
        insights["ECG"] = "ECG value is higher than normal, which may signal heart stress or arrhythmia. Further tests are recommended."
    else:
        insights["ECG"] = "ECG value is within the normal range, indicating no immediate heart irregularities."

    # Glucose Insights
    if glucose_value < 70:
        insights["Glucose"] = "Glucose level is low, which could be hypoglycemia. Ensure you are eating regularly and consult a doctor."
    elif glucose_value > 100:
        insights["Glucose"] = "Glucose level is elevated, suggesting possible prediabetes. It's advisable to monitor your blood sugar and consult a healthcare provider."
    else:
        insights["Glucose"] = "Glucose level is within the normal range. Maintain a balanced diet to keep it in check."

    return insights

# Recommendations function
def get_recommendations(ecg_value, glucose_value):
    recommendations = {}
    
    if ecg_value < 0.8:
        recommendations["ECG"] = "Consider consulting a cardiologist and undergo further heart health assessments."
    elif ecg_value > 1.0:
        recommendations["ECG"] = "Consult a healthcare provider for further cardiac testing and ensure proper lifestyle changes."
    else:
        recommendations["ECG"] = "Continue with a heart-healthy lifestyle including exercise, stress management, and proper sleep."

    if glucose_value < 70:
        recommendations["Glucose"] = "If symptoms persist, consult a doctor immediately and ensure proper nutrition."
    elif glucose_value > 100:
        recommendations["Glucose"] = "Focus on reducing sugar intake and increasing physical activity. Consider consulting a healthcare provider for more personalized advice."
    else:
        recommendations["Glucose"] = "Keep monitoring glucose levels and maintain a balanced diet and regular exercise."

    return recommendations
