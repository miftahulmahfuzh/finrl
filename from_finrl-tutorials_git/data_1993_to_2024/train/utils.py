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

    # train_days = train_df["day"].unique()[:300]
    # train_df = train_df[train_df["day"].isin(train_days)]
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
                    # non_one_values = row[ohlcv_cols][(row[ohlcv_cols] > 2) & (row[ohlcv_cols] != 0)].dropna()
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

def _temporal_variation_df(_df, _features, periods=1):
    """Calculates the temporal variation dataframe. For each feature, this
    dataframe contains the rate of the current feature's value and the last
    feature's value given a period. It's used to normalize the dataframe.

    Args:
        periods: Periods (in time indexes) to calculate temporal variation.

    Returns:
        Temporal variation dataframe.
    """
    _tic_column = "tic"
    df_temporal_variation = _df.copy()
    prev_columns = []
    for column in _features:
        prev_column = f"prev_{column}"
        prev_columns.append(prev_column)
        df_temporal_variation[prev_column] = df_temporal_variation.groupby("tic")[column].shift(periods=periods)
        df_temporal_variation[column] = (
            df_temporal_variation[column] / df_temporal_variation[prev_column]
        )
    df_temporal_variation = (
        df_temporal_variation.drop(columns=prev_columns)
        .fillna(1)
        .reset_index(drop=True)
    )
    return df_temporal_variation


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

    new_df = process_ohlcv(test_processed)
    new_df.head()
