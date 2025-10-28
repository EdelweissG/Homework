# Bass Model Forecasting: Transparent TV Adoption

## Overview
This project analyzes historical sales data of flat-screen TVs in Germany (2006–2020) and uses the **Bass Diffusion Model** to forecast adoption patterns for LG Signature OLED Transparent TVs (2025–2039). A **Fermi-adjusted approach** is applied to account for market uncertainties and adoption constraints.

---

## Repository Structure
# Bass Model Forecasting: Transparent TV Adoption

- **data/**
  - `flat_screen_sales.xlsx` — Historical sales dataset of flat-screen TVs
- **img/**
  - `sales_plot.png` — Plot of historical flat-screen TV sales
  - `prediction_plot.png` — Bass model prediction for Transparent TV
  - `fermi.png` — Fermi-adjusted forecast plot
  - `comparison_plot.png` — Comparison of flat-screen vs Transparent TV adoption
- **report/**
  - `report.pdf` — Final project report in PDF
  - `report_source.md` — Source markdown for the report
- **scripts/**
  - `script1.py` — Data preprocessing and historical sales plotting
  - `script2.py` — Bass model fitting and Transparent TV adoption forecast
  - `helper_functions.py` — Helper functions used in the project
- `README.md` — Project documentation
- `.gitignore` — Files to ignore in Git

## Setup Instructions

## Usage

### Run the scripts
- `script1.py` loads historical flat-screen TV sales data and generates a plot.  
- `script2.py` fits the Bass diffusion model, forecasts Transparent TV adoption, applies Fermi adjustments, and creates all adoption plots.  
- `helper_functions.py` contains utility functions used by the scripts.  

**Example:**

python3 scripts/script1.py
python3 scripts/script2.py

## Generated outputs
- Plots are saved automatically in the `img/` folder.  
- Forecast tables are printed in the terminal when running the scripts.  


## Data Source
- Statista (2021). Sales volume of flat-screen TVs in Germany from 2006 to 2020.  
[Link to dataset](https://www.statista.com/statistics/460190/sales-of-flat-screen-tvs-in-germany/)



** Key Findings **

- **Estimated Bass model parameters:**
  - `p` (coefficient of innovation): ~0.0493  
  - `q` (coefficient of imitation): ~0.1105  
  - `M` (market potential, millions): ~147.37  

- **Fermi-adjusted forecast (75% realistic adoption) predicts:**
  - Peak adoption year: **2030** (~6.37 million units)  
  - Five-year cumulative adoption (2025–2029): ~30.27 million units (27.4% of adjusted market potential)  
  - Cumulative adoption by 2039: ~83.4 million units  

- Transparent TV adoption follows a slower S-shaped diffusion curve compared to flat-screen TVs.
