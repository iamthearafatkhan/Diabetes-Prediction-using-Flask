# Import necessary packages
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras  # Import keras properly
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import pickle

# Load and preprocess data
file_path = "Diabetes_bd.csv"
df = pd.read_csv(file_path)

# Select features and target
selected_features = ['age', 'gender', 'pulse_rate', 'glucose', 'bmi', 'family_diabetes', 'hypertensive']
X = df[selected_features]
y = df['diabetic']

# Encode categorical variables
le = LabelEncoder()
X['gender'] = le.fit_transform(X['gender'])
y = le.fit_transform(y)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and train ANN model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Save trained ANN model
model.save('diabetes_ann_model.h5')
print("Model saved as diabetes_ann_model.h5")

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved as scaler.pkl")
