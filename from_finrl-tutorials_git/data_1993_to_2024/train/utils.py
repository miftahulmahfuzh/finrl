import pandas as pd
import numpy as np
from tqdm import tqdm

def get_quality_tickers(df):
    quality_excel = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/train/xlsx/company_quality.xlsx"
    quality_df = pd.read_excel(quality_excel, sheet_name="Company Quality")
    quality_df = quality_df[quality_df["Company Score"] == "Best"]
    tickers = quality_df["Ticker"].unique()
    # print(len(tickers))
    new_df = df[df["tic"].isin(tickers)]
    return new_df

def split_data_based_on_date(
        df: pd.DataFrame,
        TRAIN_START_DATE: str,
        TRAIN_END_DATE: str,
        DEV_START_DATE: str,
        DEV_END_DATE: str,
        TEST_START_DATE: str,
        TEST_END_DATE: str,
    ):
    # Ensure the dfdate column is in datetime format
    df['dfdate'] = pd.to_datetime(df['date'])

    # Create boolean masks for training and testing data
    train_mask = (df['dfdate'] >= TRAIN_START_DATE) & (df['dfdate'] <= TRAIN_END_DATE)
    dev_mask = (df['dfdate'] >= DEV_START_DATE) & (df['dfdate'] <= DEV_END_DATE)
    test_mask = (df['dfdate'] >= TEST_START_DATE) & (df['dfdate'] <= TEST_END_DATE)

    # Split the data
    train_df = df[train_mask].copy()
    dev_df = df[dev_mask].copy()
    test_df = df[test_mask].copy()

    train_days = train_df["day"].unique()[:300]
    train_df = train_df[train_df["day"].isin(train_days)]
    # print(train_df)

    # Validate the splits
    print("\n====================================")
    print(f"Train data range: {train_df['dfdate'].min()} to {train_df['dfdate'].max()}")
    print(f"Train data shape: {train_df.shape}")
    print(f"Dev data range: {dev_df['dfdate'].min()} to {dev_df['dfdate'].max()}")
    print(f"Dev data shape: {dev_df.shape}")
    print(f"Test data range: {test_df['dfdate'].min()} to {test_df['dfdate'].max()}")
    print(f"Test data shape: {test_df.shape}")

    return train_df, dev_df, test_df

def clean_ohlcv_data(df):
    """
    Clean OHLCV DataFrame by:
    1. Grouping by ticker symbol
    2. Filling NaN, zero, or 1 values with previous non-1/non-zero value for the same ticker

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with OHLCV data

    Returns:
    --------
    pandas.DataFrame
        Cleaned DataFrame with no NaN, zero, or 1 values
    """
    # Create a copy of the DataFrame to avoid modifying the original
    cleaned_df = df.copy()

    def clean_ticker_data(ticker_group):
        # Columns to check and fill
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        # ohlcv_cols = ["close"]

        # Create a mask for rows with problematic values
        problematic_mask = ticker_group[ohlcv_cols].apply(
            lambda col: (col.isna()) | (col == 0) | (col == 1)
        )

        # Iterate through columns to fill
        for col in ohlcv_cols:
            # Find the first non-problematic value for each column
            ticker_group[col] = ticker_group[col].replace({0: np.nan, 1: np.nan})
            # ticker_group[col] = ticker_group[col].fillna(method='ffill')
            ticker_group[col] = ticker_group[col].bfill()

        return ticker_group

    # Group by ticker and apply cleaning
    cleaned_df = cleaned_df.groupby('tic', group_keys=False).apply(clean_ticker_data)

    nan_close_counts = df.groupby('tic')['close'].apply(lambda x: x.isna().sum())

    # Print NaN close value counts
    print("Number of NaN 'close' rows for each ticker:")
    print(nan_close_counts[nan_close_counts > 0])

    # Drop any remaining rows with NaN values
    # cleaned_df = cleaned_df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])

    return cleaned_df

import pandas as pd
import numpy as np

# Define the function
def process_ohlcv(df):
    """
    Process the given OHLCV DataFrame to handle missing, zero, or '1' values.

    Args:
    - df: A pandas DataFrame containing OHLCV data.

    Returns:
    - new_df: Processed DataFrame with no NaN, zero, or '1' values in OHLCV columns.
    """
    # Define the columns to process
    # ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    ohlcv_cols = ['open', 'high', 'low', 'close']

    # Group by 'tic' and iterate over each group
    grouped = df.groupby('tic')
    for tic, group in tqdm(grouped, total=len(grouped)):
        # Iterate over each row in the group
        for index, row in group.iterrows():
            # Replace NaN, 0, or 1 values with non-1 values from the same row
            for col in ohlcv_cols:
                if row[col] in [0, 1] or pd.isna(row[col]):
                    non_one_values = row[ohlcv_cols][(row[ohlcv_cols] != 1) & (row[ohlcv_cols] != 0)].dropna()
                    if not non_one_values.empty:
                        df.at[index, col] = non_one_values.mean()  # Use the mean as a replacement

        # Replace remaining NaN, 0, or 1 values with values after in the same 'tic'
        for col in ohlcv_cols:
            group[col] = group[col].replace({0: np.nan, 1: np.nan})
            # group[col] = group[col].fillna(method='bfill')
            group[col] = group[col].bfill()

        # Update the DataFrame with processed group
        df.update(group)

    # Check and print the number of 'close' rows with NaN for each 'tic'
    # nan_counts = df.groupby('tic')['close'].apply(lambda x: x.isna().sum())
    # print("Number of 'close' rows with NaN values for each 'tic':\n", nan_counts)
    grouped = df.groupby('tic')
    nan_count_per_tic = {}
    for tic, group in tqdm(grouped, total=len(grouped)):
        nan_count = group['close'].isna().sum()
        if nan_count > 0:
            nan_count_per_tic[tic] = nan_count
    if nan_count_per_tic:
        print("Tics with NaN 'close' values:")
        for tic, count in nan_count_per_tic.items():
            print(f"{tic}: {count}")

    # Ensure no NaN, 0, or 1 values remain in the DataFrame
    df[ohlcv_cols] = df[ohlcv_cols].replace({0: np.nan, 1: np.nan})
    # df[ohlcv_cols] = df[ohlcv_cols].fillna(method='bfill').fillna(method='ffill')
    df[ohlcv_cols] = df[ohlcv_cols].bfill().ffill()
    return df

if __name__=="__main__":
    # Example usage:
    TRAIN_START_DATE = "1993-01-04"
    TRAIN_END_DATE = "2022-12-31"
    DEV_START_DATE = "2023-01-01"
    DEV_END_DATE = "2023-12-31"
    TEST_START_DATE = "2024-01-01"
    TEST_END_DATE = "2024-11-26"
    x = split_data_based_on_date(
            processed,
            TRAIN_START_DATE,
            TRAIN_END_DATE,
            DEV_START_DATE,
            DEV_END_DATE,
            TEST_START_DATE,
            TEST_END_DATE)
    train_processed, dev_processed, test_processed = x

    # cleaned_ohlcv = clean_ohlcv_data(test_processed)
    new_df = process_ohlcv(test_processed)
    new_df.head()
