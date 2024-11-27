import pandas as pd
import random
from datetime import datetime

# Function to generate random total sales for each month/branch
def generate_sales_data(year, month, branch):
    # Generate a random sales value between 100,000 and 1,000,000 for each branch
    total_sales = random.randint(100000, 1000000)
    return total_sales

# List of branches
branches = ['Kkopi', 'Waffly', 'WaterStation']

# List of months (using full month names)
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

# Prepare list to store the generated data
data = []

# Generate data for the years 2022-2024
for year in range(2022, 2025):
    for month in months:
        for branch in branches:
            total_sales = generate_sales_data(year, month, branch)
            # Append the generated row to the data list
            data.append([year, month, branch, total_sales])

# Create a DataFrame
df = pd.DataFrame(data, columns=["Year", "Month", "Branch", "Total Branch Sales"])

# Save the DataFrame to a CSV file
df.to_csv('2022_2024_sales.csv', index=False)

# Print the DataFrame to check the result
print(df.head())  # Print the first few rows to verify
