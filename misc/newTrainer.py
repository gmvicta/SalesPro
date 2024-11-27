import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Load the dataset
file_path = '2022_2024_sales.csv'  # Update with your file's location
data = pd.read_csv(file_path)

# Preprocess the data
def preprocess_data(data):
    # Encode categorical columns
    label_encoder = LabelEncoder()
    data['Branch'] = label_encoder.fit_transform(data['Branch'])  # Encode branch names

    # Create date features
    data['Month'] = pd.to_datetime(data['Month'], format='%B').dt.month  # Convert month name to numerical
    data['Year'] = data['Year'].astype(int)

    # Features and target
    X = data[['Year', 'Month', 'Branch']]
    y = data['Total Branch Sales']

    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, label_encoder, scaler

X, y, label_encoder, scaler = preprocess_data(data)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
def build_model(input_shape):
    model = Sequential([
        Dense(64, input_dim=input_shape, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')  # Linear activation for regression
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_model(X_train.shape[1])

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Save the model as .pkl
model_path = 'sales_forecasting_model.pkl'
joblib.dump(model, model_path)  # Save the trained model
print(f"Model saved to {model_path}")

# Save the label encoder and scaler
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# Function to predict new data
def predict_sales(year, month, branch):
    branch_encoded = label_encoder.transform([branch])[0]  # Encode branch
    month_num = pd.to_datetime(month, format='%B').month  # Convert month name to number
    input_data = np.array([[year, month_num, branch_encoded]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return prediction[0][0]

# Example usage
print(predict_sales(2025, 'January', 'Kkopi'))
