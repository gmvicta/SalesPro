import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)

# Load sales data
data = pd.read_csv('2022_2024_sales.csv')

# Function to preprocess data and train a model
def train_sales_model():
    # Encode the 'Month' column
    le = LabelEncoder()
    data['Month'] = le.fit_transform(data['Month'])

    # Define features (Year, Month, Branch) and target (Total Branch Sales)
    X = data[['Year', 'Month', 'Branch']]
    X['Branch'] = X['Branch'].map({'Kkopi': 0, 'Waffly': 1, 'WaterStation': 2})  # Encode Branch

    y = data['Total Branch Sales']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the trained model and LabelEncoder
    joblib.dump(model, 'sales_forecasting_model.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    return model, le

# Load the trained model and LabelEncoder (if they exist)
try:
    model = joblib.load('sales_forecasting_model.pkl')
    le = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    model, le = train_sales_model()

# Define a route for the homepage (index.html)
@app.route('/')
def home():
    return render_template('index.html')  # Renders index.html

# API route for predicting sales
@app.route('/predict_sales', methods=['POST'])
def predict_sales():
    try:
        # Get user input from JSON request
        data = request.json
        months_to_predict = int(data.get('months_to_predict', 0))  # Default to 0 if not provided
        years_to_predict = int(data.get('years_to_predict', 0))  # Default to 0 if not provided

        if not months_to_predict and not years_to_predict:
            return jsonify({'status': 'error', 'message': 'Please provide either months_to_predict or years_to_predict.'})

        # Prepare data for prediction
        monthly_predictions = []
        yearly_predictions = []
        current_year = 2025  # Starting year
        current_month = 12   # Assuming we're starting from November

        # Generate monthly predictions for all branches if months_to_predict is provided
        if months_to_predict > 0:
            for branch_value in [0, 1, 2]:  # Loop through all branches (Kkopi, Waffly, WaterStation)
                for i in range(months_to_predict):
                    month = (current_month + i) % 12
                    year = current_year + (current_month + i) // 12
                    feature = np.array([[year, month, branch_value]])
                    predicted_sales = model.predict(feature)[0]
                    monthly_predictions.append({
                        'Year': year,
                        'Month': month,
                        'Branch': ['Kkopi', 'Waffly', 'WaterStation'][branch_value],
                        'Predicted Sales': predicted_sales
                    })

        # Generate yearly predictions for all branches if years_to_predict is provided
        if years_to_predict > 0:
            for branch_value in [0, 1, 2]:  # Loop through all branches (Kkopi, Waffly, WaterStation)
                for i in range(years_to_predict):
                    year = current_year + i
                    # Sum the monthly sales for each year
                    yearly_sales = 0
                    for month in range(12):  # Iterate over all months of the year
                        feature = np.array([[year, month, branch_value]])
                        predicted_sales = model.predict(feature)[0]
                        yearly_sales += predicted_sales  # Add up the predicted sales for each month

                    # Store the yearly prediction as the sum of all months
                    yearly_predictions.append({
                        'Year': year,
                        'Branch': ['Kkopi', 'Waffly', 'WaterStation'][branch_value],
                        'Predicted Sales': yearly_sales
                    })

        # Sort and process predictions
        monthly_predictions.sort(key=lambda x: (x['Year'], x['Month']))
        month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        for prediction in monthly_predictions:
            prediction['Month'] = month_names[prediction['Month']]

        # Return separate predictions
        return jsonify({
            'status': 'success',
            'monthly_predictions': monthly_predictions,
            'yearly_predictions': yearly_predictions
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
    
if __name__ == '__main__':
    app.run(debug=True)
