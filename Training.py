import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import joblib

# ✅ STEP 1: Load & Preprocess the Dataset
# Load Pima Indians Diabetes dataset (Glucose Levels) + Synthetic ECG Data
df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
                 names=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigree", "Age", "Label"])

# ✅ Simulate ECG readings (synthetic, normally distributed)
np.random.seed(42)
df["ECG"] = np.random.normal(loc=0.8, scale=0.2, size=len(df))  # ECG values (simulated, mean 0.8)

# Select relevant columns
X = df[['ECG', 'Glucose']].values  # Features (ECG + Glucose)
y = df['Label'].values  # Labels (0 = Normal, 1 = High-Risk)

# ✅ STEP 2: Split Data for Training & Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ STEP 3: Normalize the Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for API preprocessing
joblib.dump(scaler, "scaler.pkl")

# ✅ STEP 4: Define & Train a Neural Network
model = Sequential([
    Input(shape=(2,)),  # Two input features: ECG & Glucose
    Dense(32, activation="relu"),  # Increase number of neurons for better learning
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")  # Output layer (Binary Classification)
])

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# ✅ STEP 5: Evaluate & Save the Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"✅ Final Model Accuracy: {accuracy * 100:.2f}%")

# Save the model for the API
model.save("health_model1.keras")

# ✅ STEP 6: Convert Model to TFLite (for Android & API)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite Model
with open("health_model1.tflite", "wb") as f:
    f.write(tflite_model)

print("🚀 Model trained & converted to TFLite successfully!")
