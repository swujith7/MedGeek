import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import matplotlib.pyplot as plt


# Load dataset
df = pd.read_csv("health_data.csv")
X = df[['ECG', 'Glucose']].values
y = df['Label'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert labels if necessary (for classification tasks)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# Define the neural network model 2
model = keras.Sequential([
    keras.layers.Input(shape=(2,)),  # ✅ Corrected Input Layer
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Binary classification output
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_scaled, y_train, epochs=50, validation_data=(X_test_scaled, y_test), verbose=1)
test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
print(f"✅ Model Accuracy: {test_acc * 100:.2f}%")

# ✅ Save Model
model.save("ecg_glucose_risk_model.keras")
model.save("health_risk_model.h5")
# ✅ Convert Model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# ✅ Save TFLite Model
tflite_model_path = "ecg_glucose_risk_model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"✅ Model successfully converted to TensorFlow Lite: {tflite_model_path}")

# ✅ Plot Training Performance
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title('Model Accuracy Over Epochs')
plt.legend()
plt.show()
