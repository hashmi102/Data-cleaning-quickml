# Data Cleaning & Quick-ML Dashboard

Interactive Streamlit app for cleaning tabular data, performing compact exploratory analysis, and training simple ML models (Linear, Ridge, Lasso, Logistic, Decision Trees).  
Designed for data practitioners and non-technical users to quickly go from raw CSV → cleaned dataset → model evaluation.

## Features
- Upload CSV and preview first 200 rows.
- Missing value analysis and drop-by-threshold.
- Per-column imputation (mean / median / most_frequent / constant).
- Categorical encoding (Label, One-Hot).
- Numeric scaling (Standard, MinMax).
- Compact EDA: small histograms, bar charts, correlation heatmap.
- Train/test split, choose model and hyperparameters, view metrics, small plots.
- Download cleaned dataset and test predictions as CSV.

## Quick start (test locally)
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv venv
# activate venv (Windows)
venv\Scripts\activate
# (macOS / Linux)
source venv/bin/activate
pip install -r requirements.txt
streamlit run data_cleaning_app.py
