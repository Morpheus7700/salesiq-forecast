# salesiq-forecast

> A multi-level sales-forecasting script built on Facebook Prophet.

A Python utility that reads aggregated monthly sales data and produces forecasts at several levels of granularity — overall, and broken down by branch, product, sales rep, and customer — using Prophet.

## Features
- Forecasts at multiple levels from a single run (overall + grouped breakdowns)
- Configurable forecast horizon (default 3 months)
- Handles seasonality automatically via Prophet
- Writes results to `output/sales_forecast.csv`

## Tech Stack
- Python, Pandas, Prophet

## Usage
```bash
pip install prophet pandas
# place your aggregated data as input_sales.csv, then:
python forecast.py
```
