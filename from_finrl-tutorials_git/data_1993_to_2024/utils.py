import pandas as pd
import numpy as np

def clean_ohlcv(df):
    """
    Clean OHLCV (Open, High, Low, Close, Volume) data by replacing NaN or zero values with 1.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing financial data

    Returns:
    --------
    pandas.DataFrame
        Cleaned DataFrame with NaN and zero values replaced
    """
    # Columns to check and replace
    columns_to_check = ['open', 'high', 'low', 'close', 'volume']

    # Create a copy of the DataFrame to avoid modifying the original
    cleaned_df = df.copy()

    # Replace NaN or zero values with 1 for specified columns
    for col in columns_to_check:
        cleaned_df[col] = cleaned_df[col].apply(lambda x: 1 if pd.isna(x) or x == 0 else x)

    return cleaned_df

def add_needed_dates(dfi, needed_dates, tic):
    """
    Add dummy rows for missing dates in the DataFrame.

    Parameters:
    -----------
    dfi : pandas.DataFrame
        Input DataFrame with stock data
    needed_dates : list
        List of all dates that should be present in the DataFrame
    tic : str
        Ticker symbol for the stock

    Returns:
    --------
    pandas.DataFrame
        DataFrame with added dummy rows for missing dates
    """
    # Convert needed_dates to a set for faster lookups
    needed_dates_set = set(needed_dates)

    # Convert the 'date' column to datetime
    # dfi['date'] = pd.to_datetime(dfi['date'])

    # Create a set of existing dates in the DataFrame
    existing_dates_set = set(dfi['date'])

    # Find missing dates
    missing_dates = needed_dates_set - existing_dates_set

    # Create dummy rows for missing dates
    dummy_rows = []
    for dx in missing_dates:
        dummy_row = {
            # 'date': pd.to_datetime(dx),
            'date': dx,
            'open': 1,
            'high': 1,
            'low': 1,
            'close': 1,
            'volume': 1,
            'tic': tic
        }
        dummy_rows.append(dummy_row)

    # Add dummy rows to the DataFrame
    if dummy_rows:
        dummy_df = pd.DataFrame(dummy_rows)
        dfi = pd.concat([dfi, dummy_df], ignore_index=True)

    dfi = dfi[dfi["date"].isin(needed_dates)]

    # Sort the DataFrame by date
    dfi = dfi.sort_values('date')

    # Drop the existing 'day' column if it exists
    if 'day' in dfi.columns:
        dfi = dfi.drop(columns=['day'])

    # Create the 'day' column using the date2day mapping
    date2day = {date: day for day, date in enumerate(needed_dates)}
    dfi['day'] = dfi['date'].map(date2day)

    return dfi

def validate_tic_dates(all_df):
    """
    Validates that each ticker (tic) in the DataFrame has the same unique dates.

    Steps:
    1. Counts the unique dates for each ticker and saves it in a dictionary.
    2. Asserts that all tickers have the same unique dates and the same total unique dates.

    Parameters:
    - all_df (pd.DataFrame): The DataFrame containing stock data with 'date' and 'tic' columns.

    Returns:
    - dict: A dictionary with tickers as keys and their total unique dates as values.
    - int: Total number of unique dates across the dataset.

    Raises:
    - AssertionError: If any ticker has a different set or total count of unique dates.
    """
    # Group the data by 'tic' and count unique dates for each tic
    unique_dates_per_tic = all_df.groupby("tic")["date"].nunique().to_dict()

    # Check that all tickers have the same total unique dates
    total_unique_dates = len(all_df["date"].unique())
    assert all(val == total_unique_dates for val in unique_dates_per_tic.values()), \
        "Not all tickers have the same number of unique dates!"

    # Check that all tickers have the same set of dates
    unique_date_sets = all_df.groupby("tic")["date"].apply(lambda x: set(x)).to_dict()

    # Ensure that all tickers have the same set of dates
    date_sets = list(unique_date_sets.values())
    assert all(date_set == date_sets[0] for date_set in date_sets), \
        "Not all tickers share the same set of dates!"

    # Return the results
    return unique_dates_per_tic, total_unique_dates

if __name__=="__main__":
    # Example usage:
    new_total_for_each_tic, total_unique_dates = validate_tic_dates(all_df)
    print("Unique dates per ticker:", new_total_for_each_tic)
    print("Total unique dates in the dataset:", total_unique_dates)
