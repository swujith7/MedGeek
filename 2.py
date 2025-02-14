# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Step 1: Load the dataset
df = pd.read_csv("health_data.csv")  # Ensure the file exists

# Step 2: Prepare features (X) and target labels (y)
X = df[['ECG', 'Glucose']].values  # Input features
y = df['Label'].values  # Target (0 = Normal, 1 = Abnormal)

# Step 3: Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Train a Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Evaluate Model Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Step 7: Save Scaler & Model for Android
joblib.dump(scaler, "scaler.pkl")  # Save the scaler for preprocessing

# Convert to TensorFlow Model for TFLite Conversion (Fixed IndexError)
keras_model = Sequential([
    Input(shape=(2,)),  # Define Input Layer
    Dense(1, activation="sigmoid")  # Single Dense Layer
])
keras_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ✅ FIX: Set weights on the **first** (and only) layer
keras_model.layers[0].set_weights([model.coef_.T, model.intercept_])

# Save in the correct format
keras_model.save("health_model.keras")  # Use `.keras` instead of `.h5`

# Step 8: Convert Model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

# Save TFLite Model
with open("health_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model successfully trained and converted to TFLite without warnings!")
