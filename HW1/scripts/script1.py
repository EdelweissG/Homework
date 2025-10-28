import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Create folder for images if it doesn't exist
if not os.path.exists('../img'):
    os.makedirs('../img')

# Load historical flat-screen TV sales data
file_path = "../data/flat_screen_sales.xlsx"
tv_data = pd.read_excel(file_path, sheet_name='Data', skiprows=4, usecols=[1, 2], names=['Year', 'Unit Sales'])
print(tv_data.head())

# Extract data
years = tv_data['Year'].values
annual_sales = tv_data['Unit Sales'].values
cumulative_sales = np.cumsum(annual_sales)
t = years - years.min()

# Plot historical sales
plt.figure(figsize=(10, 6))
plt.plot(tv_data['Year'], tv_data['Unit Sales'], marker='o')
plt.title('Historical Flat-Screen TV Sales in Germany')
plt.xlabel('Year')
plt.ylabel('Units Sold (millions)')
plt.grid(True)
plt.savefig('../img/sales_plot.png')
plt.show()
