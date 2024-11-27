import pandas as pd
import random
from datetime import datetime, timedelta

# Define the sales data template
branches = ["Kkopi", "Waffly", "WaterStation"]
products = {
    "Kkopi": [("Coffee", "Solo"), ("Milktea", "Jumbo"), ("Fruit Tea", "Jumbo")],
    "Waffly": [
        ("Chocolate", "Regular"),
        ("Strawberry", "Regular"),
        ("Caramel", "Regular"),
        ("Choco-berry", "Regular"),
        ("Choco-caramel", "Regular"),
        ("Berry-caramel", "Regular"),
        ("Chocolate", "Large"),
        ("Strawberry", "Large"),
        ("Caramel", "Large"),
        ("Choco-berry", "Large"),
        ("Choco-caramel", "Large"),
        ("Berry-caramel", "Large"),
    ],
    "WaterStation": [
        ("Gallon Round", "New"),
        ("Gallon Round", "Refill"),
        ("Gallon Slim", "New"),
        ("Gallon Slim", "Refill"),
        ("350ml", "New"),
        ("500ml", "New"),
    ],
}
price_ranges = {
    "Kkopi": {"Solo": 39, "Jumbo": 49},
    "Waffly": {"Regular": 29, "Large": 39},
    "WaterStation": {"New": 150, "Refill": 20, "350ml": 10, "500ml": 15},
}

# Function to generate random sales data
def generate_sales_data_for_month(year, month):
    data = []
    # Get the first and last days of the month
    start_date = datetime(year, month, 1)
    end_date = (start_date + timedelta(days=31)).replace(day=1) - timedelta(days=1)
    date = start_date

    while date <= end_date:
        for branch, items in products.items():
            for product, size in items:
                quantity_sold = random.randint(2, 100)
                # Fetch price based on branch and size
                if branch == "Kkopi":
                    price = price_ranges[branch][size]  # Lookup fixed price
                elif branch == "Waffly":
                    price = price_ranges[branch][size]
                else:  # For WaterStation
                    price = price_ranges[branch][size]
                total_sales = quantity_sold * price
                data.append(
                    {
                        "Date": date.strftime("%Y-%m-%d"),
                        "Branch": branch,
                        "Product": product,
                        "Size": size,
                        "Quantity Sold": quantity_sold,
                        "Price": price,
                        "Total Sales": total_sales,
                    }
                )
        date += timedelta(days=1)
    return data

# Generate data for October 2024
sales_data = generate_sales_data_for_month(2024, 12)

# Create a DataFrame and save to CSV
df = pd.DataFrame(sales_data)
df.to_csv("sales_data.csv", index=False)

print("October sales data CSV generated as 'sales_data.csv'")
