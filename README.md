
# E-commerce Sales Optimization Engine (Streamlit)

A Streamlit dashboard built on the Sample Supermarket dataset to analyze sales, discounting, product mix, customer segments, regions, shipping mode, and profitability.

## What is included
- `app.py` - Streamlit dashboard
- `requirements.txt` - Python dependencies
- `data/E-commerce Sales Optimization Engine.xlsx` - original dataset
- `data/cleaned_supermarket_data.csv` - cleaned and transformed dataset used in the app

## Core dashboard sections
1. Business Objective & Strategy  
2. Data Audit  
3. Commercial Performance  
4. Discount & Margin Diagnostics  
5. Correlation & Driver Analysis  
6. Product Mix & Portfolio Risk  
7. Predictive Signals  
8. Recommendations  

## Data cleaning performed
- Removed exact duplicate rows
- Converted `Postal Code` from numeric to string
- Created `Profit Margin %`
- Created `Profitability Status`
- Created `Discount Band`
- Created `Sales Band`

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Upload this full project folder to a GitHub repository.
2. In Streamlit Community Cloud, create a new app from that repository.
3. Set the main file path to `app.py`.
4. Deploy.

## Notes
- The dashboard is filter-driven from the sidebar.
- The main analytical focus is descriptive and diagnostic analysis.
- A predictive section is included as a supporting extension, not as the primary objective.
