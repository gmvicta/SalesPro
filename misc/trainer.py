import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# MAIN FUNCTION
def load_and_combine_data(dataset_folder):
    all_data = []
    for month_folder in os.listdir(dataset_folder):
        month_path = os.path.join(dataset_folder, month_folder)
        if os.path.isdir(month_path):
            sales_file = os.path.join(month_path, "sales_data.csv")
            if os.path.exists(sales_file):
                print(f"Loading data from: {sales_file}")
                df = pd.read_csv(sales_file)
                if not df.empty:
                    df['Month'] = month_folder
                    all_data.append(df)
                else:
                    print(f"Warning: {sales_file} is empty.")
            else:
                print(f"Warning: {sales_file} does not exist.")
    if not all_data:
        print("No data found in the dataset folder.")
    combined_data = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    return combined_data

# LOAD & COMBINE
dataset_folder = 'dataset'  # FOLDER PATH
df = load_and_combine_data(dataset_folder)

if df.empty:
    print("No valid data loaded. Exiting script.")
    exit()

# DARA PRE-PROCESSING
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.weekday
df = pd.get_dummies(df, columns=['Branch', 'Product', 'Size'], drop_first=True)

# TARGET VAR (TOTAL SALES)
X = df.drop(columns=['Total Sales', 'Date'])
y = df['Total Sales'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MODEL ARCHITECTURE
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear')) 

# MODEL COMPILER
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# TRAINER
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
model.save("sales_prediction_model.keras")

print("Model trained and saved as 'sales_prediction_model.keras'")

# PERCENTAGE RESULT
from sklearn.metrics import r2_score
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R^2 Score: {r2:.4f}")

