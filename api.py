from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
try:
    model = tf.keras.models.load_model("health_risk_model.h5")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Define request model
class HealthData(BaseModel):
    ecg_value: float
    glucose_value: float

# Define prediction endpoint
@app.post("/predict")
def predict_health(data: HealthData):
    try:
        # Convert input data to NumPy array for model prediction
        input_data = np.array([[data.ecg_value, data.glucose_value]])
        
        # Make a prediction
        prediction = model.predict(input_data)
        probability_score = float(prediction[0][0])  # Convert model output to float

        # Determine risk level based on model output
        risk_level = "Low"
        if probability_score >= 0.8:
            risk_level = "High"
        elif probability_score >= 0.5:
            risk_level = "Moderate"

        # Deep insights based on ECG and glucose levels
        insights = {
            "Low": {
                "general": "Your ECG and glucose levels are within a normal range. This suggests a healthy cardiovascular and metabolic system.",
                "possible_conditions": ["No immediate concerns"],
                "recommendations": [
                    "Maintain a balanced diet and regular exercise.",
                    "Get an annual health checkup to track your vitals.",
                    "Stay hydrated and monitor stress levels."
                ]
            },
            "Moderate": {
                "general": "Your ECG or glucose readings indicate some irregularities. This could be an early warning sign of potential issues.",
                "possible_conditions": [
                    "Early signs of arrhythmia (if ECG is irregular)",
                    "Prediabetes (if glucose is elevated)",
                    "Electrolyte imbalance (if ECG is fluctuating)"
                ],
                "recommendations": [
                    "Monitor ECG and glucose levels regularly.",
                    "Reduce processed sugars and increase fiber intake.",
                    "Consult a general physician for further tests."
                ]
            },
            "High": {
                "general": "Your readings suggest a significant risk of cardiovascular or metabolic disorders. Immediate medical attention is recommended.",
                "possible_conditions": [
                    "Atrial fibrillation (if ECG is highly irregular)",
                    "Type 2 Diabetes (if glucose is significantly high)",
                    "Risk of cardiac arrest or stroke (if ECG and glucose are both abnormal)"
                ],
                "recommendations": [
                    "Seek urgent consultation with a cardiologist and endocrinologist.",
                    "Undergo a full blood panel test, including HbA1c and lipid profile.",
                    "Avoid high sodium and high sugar foods immediately.",
                    "Consider wearing a continuous glucose monitor (CGM) or a portable ECG device."
                ]
            }
        }

        # Return structured response
        return {
            "message": "Prediction successful",
            "input": data.dict(),
            "predicted_risk": risk_level,
            "probability_score": probability_score,
            "detailed_insights": insights[risk_level]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {e}")

# Root endpoint
@app.get("/")
def home():
    return {"message": "Health Risk Prediction API is running"}
