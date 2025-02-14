import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt


# ✅ Load the trained model
model = tf.keras.models.load_model("health_model1.keras")

# ✅ Load the scaler for preprocessing
scaler = joblib.load("scaler.pkl")

test_cases = [
    [0.7, 85],   # Normal case
    [1.5, 200],  # High risk case (High ECG & glucose)
    [1.0, 180],  # Likely diabetic case
]

for test in test_cases:
    test_scaled = scaler.transform([test])
    pred = model.predict(test_scaled)
    risk = "High Risk" if pred[0][0] > 0.5 else "Normal"
    print(f"ECG: {test[0]}, Glucose: {test[1]} → Risk: {risk}, Score: {pred[0][0]:.4f}")
history = model.history.history  # Get training history

plt.plot(history['loss'], label="Loss")
plt.plot(history['accuracy'], label="Accuracy")
plt.legend()
plt.title("Model Training Performance")
plt.show()
