# Homework 3: Survival Analysis - Telecom Customer Churn

## Overview
This homework focuses on analyzing customer churn in a telecom dataset using **survival analysis** methods. The goal is to understand which factors affect customer retention, estimate **Customer Lifetime Value (CLV)**, and identify high-risk and high-value customers for potential retention strategies.

## Files in HW3
- `survival_analysis.ipynb` : Jupyter notebook containing all code, exploratory analysis, survival models, and CLV calculations.
- `telco.csv` : Dataset of 1000 telecom customers, including demographic and service-related features.
- `requirements.txt` : Python packages needed to run the analysis.
- `Report.pdf` : Written report summarizing analysis, results, and recommendations.

## Key Analysis Steps
1. **Data Preparation**: Cleaned the dataset and encoded categorical variables.
2. **Exploratory Analysis**: Visualized distributions of tenure, age, and churn across different customer categories.
3. **Survival Modeling**: Tested three parametric models (Weibull, LogNormal, LogLogistic) using the Accelerated Failure Time (AFT) approach. The **LogNormal AFT model** provided the best fit based on AIC.
4. **Customer Lifetime Value (CLV)**: Estimated for each customer based on survival probabilities and average monthly revenue.
5. **12-Month Churn Probability**: Calculated churn risk for each customer to identify high-risk segments.
6. **Retention Insights**: Recommended retention strategies based on high-value and high-risk customers, with a suggested annual retention budget.

## Results Highlights
- Most valuable segments: **Plus service** and **E-service** customers.
- Average CLV: `$1,335.51`
- Top 25% high-value customers: 250
- Top 25% high-risk customers: 250
- High-value at-risk overlap: 0 customers
- Suggested annual retention budget: `$6,168.05`

## How to Run
1. Create a new Python environment and install dependencies:
   ```bash
   pip install -r requirements.txt

2. Open survival_analysis.ipynb in Jupyter Notebook or JupyterLab.
3. Run all cells to reproduce the analysis and figures.
