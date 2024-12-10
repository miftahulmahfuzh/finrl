import pandas as pd

def split_data_based_on_date(df, TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE):
    """
    Split the DataFrame into training and testing sets based on specified date ranges.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing a 'dfdate' column with date information
    TRAIN_START_DATE : str
        Start date for training data (inclusive)
    TRAIN_END_DATE : str
        End date for training data (inclusive)
    TEST_START_DATE : str
        Start date for testing data (inclusive)
    TEST_END_DATE : str
        End date for testing data (inclusive)

    Returns:
    --------
    train_df : pandas.DataFrame
        DataFrame containing training data
    test_df : pandas.DataFrame
        DataFrame containing testing data
    """
    # Ensure the dfdate column is in datetime format
    df['dfdate'] = pd.to_datetime(df['date'])

    # Create boolean masks for training and testing data
    train_mask = (df['dfdate'] >= TRAIN_START_DATE) & (df['dfdate'] <= TRAIN_END_DATE)
    test_mask = (df['dfdate'] >= TEST_START_DATE) & (df['dfdate'] <= TEST_END_DATE)

    # Split the data
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    # Validate the splits
    print(f"Training data range: {train_df['dfdate'].min()} to {train_df['dfdate'].max()}")
    print(f"Training data shape: {train_df.shape}")
    print(f"\nTesting data range: {test_df['dfdate'].min()} to {test_df['dfdate'].max()}")
    print(f"Testing data shape: {test_df.shape}")

    return train_df, test_df

if __name__=="__main__":
    # Example usage:
    TRAIN_START_DATE = "1993-01-04"
    TRAIN_END_DATE = "2022-12-31"
    TEST_START_DATE = "2023-01-01"
    TEST_END_DATE = "2024-11-26"
    train_processed, test_processed = split_data_based_on_date(processed, TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE)
