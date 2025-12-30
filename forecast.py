
import pandas as pd
from prophet import Prophet
import os
import subprocess
from datetime import datetime
import itertools

# --- 1. Configuration ---
INPUT_CSV = 'input_sales.csv'
OUTPUT_DIR = 'output'
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'sales_forecast.csv')
FORECAST_PERIOD_MONTHS = 3

# Define the different levels of granularity for forecasting.
# The script will run a forecast for each unique combination of the columns in each list.
FORECAST_LEVELS = {
    'Overall': [], # Add an overall forecast
    'By_BCode': ['BCode'],
    'By_BCode_Product': ['BCode', 'ProductCode'],
    'By_BCode_PSR': ['BCode', 'PSRCode'],
    'By_BCode_Customer': ['BCode', 'CustomerName']
}

# --- 2. Main Execution ---
all_forecasts = []
print("--- Sales Forecast Script Started ---")

try:
    print(f"Reading aggregated sales data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # Standardize column names for Prophet
    df.rename(columns={'MonthStart': 'ds', 'NET_TRADE_AMOUNT_CR': 'y'}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds'])

    # Get a list of all possible key columns to ensure they are added to the final DataFrame
    all_key_cols_flat = sorted(list(set(itertools.chain.from_iterable(FORECAST_LEVELS.values()))))

    # Loop through each defined forecast level (e.g., 'By_BCode', 'By_BCode_Product', etc.)
    for level_name, group_by_cols in FORECAST_LEVELS.items():
        print(f"\n--- Processing forecast level: {level_name} ---")
        
        current_df = df.copy()

        if group_by_cols: # If there are grouping columns (not 'Overall' forecast)
            # Drop rows where any of the grouping keys are null, as they can't be grouped.
            current_df = current_df.dropna(subset=group_by_cols)
            if current_df.empty:
                print(f"  No data for grouping columns: {group_by_cols}. Skipping this level.")
                continue
            grouped = current_df.groupby(group_by_cols)
            num_groups = len(grouped)
            print(f"Found {num_groups} unique combinations to forecast.")

            # Loop through each unique combination (e.g., each specific BCode, or each BCode-ProductCode pair)
            for i, (group_keys, group_df) in enumerate(grouped):
                # Provide progress update
                if num_groups > 1: # Only show progress if there's more than one group
                    print(f"  - Forecasting for group {i+1} of {num_groups} ({group_keys})...", end='\r')

                # Prophet needs at least 2 data points
                if len(group_df) < 2:
                    continue

                # Ensure data is sorted by date for Prophet
                prophet_df = group_df[['ds', 'y']].sort_values('ds')

                # --- Model Training and Forecasting ---
                model = Prophet(changepoint_prior_scale=0.5)
                model.fit(prophet_df)
                future = model.make_future_dataframe(periods=FORECAST_PERIOD_MONTHS, freq='MS')
                forecast = model.predict(future)
                
                # --- Prepare Output for this specific forecast ---
                forecast['ForecastLevel'] = level_name
                
                # Add the group key columns (e.g., the values of 'BCode', 'ProductCode') to the output
                if not isinstance(group_keys, tuple):
                    group_keys = (group_keys,) # Ensure group_keys is always a tuple
                for j, col_name in enumerate(group_by_cols):
                    forecast[col_name] = group_keys[j]

                # Merge actuals from the past into the forecast data
                forecast = forecast.merge(prophet_df.rename(columns={'y': 'Actual'}), on='ds', how='left')
                
                all_forecasts.append(forecast)
            print(f"\nCompleted {num_groups} forecasts for level {level_name}.")

        else: # Handle 'Overall' forecast (no grouping)
            print("  Forecasting overall sales (no grouping).")
            if len(current_df) < 2:
                print("  Not enough data for overall forecast. Skipping.")
                continue

            prophet_df = current_df[['ds', 'y']].sort_values('ds')
            model = Prophet(changepoint_prior_scale=0.5)
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=FORECAST_PERIOD_MONTHS, freq='MS')
            forecast = model.predict(future)

            forecast['ForecastLevel'] = level_name
            forecast = forecast.merge(prophet_df.rename(columns={'y': 'Actual'}), on='ds', how='left')
            all_forecasts.append(forecast)
            print("Completed overall forecast.")

    if not all_forecasts:
        raise ValueError("No forecasts were generated. Check input data and FORECAST_LEVELS configuration.")

    # --- 3. Combine, Clean, and Save Final Output ---
    print("\nCombining all forecasts into a single file...")
    final_df = pd.concat(all_forecasts, ignore_index=True)
    
    # Rename for Power BI
    final_df.rename(columns={'ds': 'Date', 'yhat': 'Forecast', 'yhat_lower': 'Forecast_Low', 'yhat_upper': 'Forecast_High'}, inplace=True)
    final_df['Date'] = pd.to_datetime(final_df['Date']).dt.strftime('%Y-%m-%d')

    # Ensure all key columns exist in the final dataframe, filling missing with None/NaN
    for col in all_key_cols_flat:
        if col not in final_df.columns:
            final_df[col] = pd.NA # Use pd.NA for nullable types

    # Define the final column order explicitly
    final_output_cols = ['Date', 'ForecastLevel'] + all_key_cols_flat + ['Actual', 'Forecast', 'Forecast_Low', 'Forecast_High']
    final_df = final_df[final_output_cols]

    # Save to CSV
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_DIR), exist_ok=True)
    final_df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_CSV), index=False)
    print(f"Successfully saved combined forecast to {OUTPUT_CSV}")


    # --- 4. Push to GitHub ---
    print("\nAttempting to push forecast to GitHub...")
    # Change current working directory to the script's directory for git commands
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    try:
        # Add the specific output file to staging
        subprocess.run(["git", "add", OUTPUT_CSV], check=True, capture_output=True, text=True)
        commit_message = f"Auto-update sales forecast: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Check git status to see if there are changes to commit
        status_result = subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True)
        if OUTPUT_CSV.replace('\\', '/') in status_result.stdout.replace('\\', '/'): # Normalize paths for comparison
            subprocess.run(["git", "commit", "-m", commit_message], check=True, capture_output=True, text=True)
            print("Committed updated forecast.")
            
            print("Pushing to remote repository...")
            subprocess.run(["git", "push"], check=True, capture_output=True, text=True)
            print("Successfully pushed forecast to GitHub.")
        else:
            print("No changes in forecast file to commit. Nothing to push.")

    except FileNotFoundError:
        print("\n--- GIT ACTION FAILED ---")
        print("Error: 'git' command not found. Please ensure Git is installed and in your system's PATH.")
    except subprocess.CalledProcessError as e:
        print("\n--- GIT ACTION FAILED ---")
        print(f"An error occurred during Git operation.\nStderr: {e.stderr}\nStdout: {e.stdout}")
    finally:
        # Change back to the original working directory if necessary (though not critical for this agent)
        pass # The agent context manages this


except Exception as e:
    print(f"\n--- SCRIPT FAILED ---")
    print(f"An error occurred: {e}")
    # In case of an error, this helps in debugging
    import traceback
    traceback.print_exc()

print("\n--- Sales Forecast Script Finished ---")
