import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from helper_functions import bass_model, bass_cumulative, bass_yearly
import os

# Load historical sales
file_path = "../data/flat_screen_sales.xlsx"
tv_data = pd.read_excel(file_path, sheet_name='Data', skiprows=4, usecols=[1, 2], names=['Year', 'Unit Sales'])

years_hist = tv_data['Year'].values
annual_sales = tv_data['Unit Sales'].values
cumulative_sales = np.cumsum(annual_sales)
t_hist = years_hist - years_hist.min()

# Fit Bass model
initial_guess = [0.03, 0.4, 120]  # p, q, M
params, _ = curve_fit(bass_model, t_hist, cumulative_sales, p0=initial_guess, bounds=(0, [1,1,1000]))
p, q, M = params

print("\nEstimated Bass Model Parameters:")
print(f"p (coefficient of innovation): {p:.4f}")
print(f"q (coefficient of imitation): {q:.4f}")
print(f"M (market potential, millions): {M:.2f}")

# Forecast for Transparent TV
years_forecast = np.arange(2025, 2040)
t_forecast = np.arange(1, len(years_forecast)+1)

yearly_adoption = bass_yearly(t_forecast, p, q, M)
cumulative_adoption = bass_cumulative(t_forecast, p, q, M)

# Fermi adjustment
fermi_factor = 0.75
yearly_adoption_fermi = yearly_adoption * fermi_factor
cumulative_adoption_fermi = np.cumsum(yearly_adoption_fermi)

forecast_df_fermi = pd.DataFrame({
    "Year": years_forecast,
    "Yearly Adoption (millions)": yearly_adoption_fermi,
    "Cumulative Adoption (millions)": cumulative_adoption_fermi
})

# Save forecast CSV
if not os.path.exists('../data'):
    os.makedirs('../data')
forecast_df_fermi.to_csv('../data/forecast_transparent_tv.csv', index=False)

print("\nEstimated Number of Adopters by Period for Transparent TV (Fermi-adjusted):")
print(forecast_df_fermi.round(2))

# Key insights
peak_index = np.argmax(yearly_adoption_fermi)
peak_year = forecast_df_fermi['Year'][peak_index]
peak_value = yearly_adoption_fermi[peak_index]
five_year_adoption = forecast_df_fermi['Yearly Adoption (millions)'][:5].sum()
five_year_percentage = (five_year_adoption / (M * fermi_factor)) * 100

print(f"\nKey Adoption Insights for Transparent TV:")
print(f"Peak Adoption Year: {peak_year} with {peak_value:.2f} million units")
print(f"Five-Year Cumulative Adoption: {five_year_adoption:.2f} million units ({five_year_percentage:.1f}% of adjusted market potential)")

# Plot Fermi-adjusted adoption
plt.figure(figsize=(12, 6))
plt.plot(forecast_df_fermi['Year'], forecast_df_fermi['Yearly Adoption (millions)'], marker='o', color='green', label='Yearly Adoption')
plt.plot(forecast_df_fermi['Year'], forecast_df_fermi['Cumulative Adoption (millions)'], marker='s', color='purple', label='Cumulative Adoption')
plt.title('Fermi-Adjusted Adoption Forecast: Transparent TV')
plt.xlabel('Year')
plt.ylabel('Number of Adopters (millions)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('../img/fermi.png')
plt.show()

# Compare with flat-screen TVs
years_flat = tv_data['Year']
adoption_flat = tv_data['Unit Sales'].cumsum()
years_transparent = forecast_df_fermi['Year']
adoption_transparent = forecast_df_fermi['Cumulative Adoption (millions)']

plt.figure(figsize=(10,6))
plt.plot(years_flat, adoption_flat, 'o-', label="Flat-Screen TVs (Historical)")
plt.plot(years_transparent, adoption_transparent, 's--', label="Transparent TVs (Forecast)")
plt.title("Adoption Curves: Flat-Screen vs Transparent TVs")
plt.xlabel("Year")
plt.ylabel("Cumulative Adoption (millions)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('../img/comparison_plot.png')
plt.show()

