import pandas as pd
import numpy as np

def sliding_windows_normalization(df, features=["close", "high", "low", "volume"], window_size=3):
    """
    Apply Sliding Window Min-Max Normalization to specified features.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with time series data
    features : list, optional
        List of column names to normalize (default: ["close", "high", "low", "volume"])
    window_size : int, optional
        Size of the sliding window for normalization (default: 3)

    Returns:
    --------
    normalized_df : pandas.DataFrame
        DataFrame with normalized features, first (window_size-1) rows removed for each ticker
    """
    # Create a list to store normalized DataFrames for each ticker
    normalized_dfs = []

    # Group by ticker and process each group
    for tic, group in df.groupby('tic'):
        # Create a copy of the group to avoid modifying the original
        group_normalized = group.copy()

        # Iterate through specified features
        for feature in features:
            # Ensure the feature exists in the dataframe
            if feature not in group.columns:
                raise ValueError(f"Feature {feature} not found in the dataframe")

            # Create a list to store normalized values
            normalized_values = []

            # Iterate through each row in the group
            for t in range(len(group)):
                # Determine the window range
                start = max(0, t - window_size + 1)
                window = group[feature].iloc[start:t+1]

                # Handle case when window is empty or has constant values
                if len(window) == 0:
                    normalized_values.append(np.nan)
                elif (window.min() == window.max()):
                    # normalized_values.append(0.0)
                    normalized_values.append(1)
                else:
                    # Apply sliding window min-max normalization
                    current_value = group.loc[group.index[t], feature]
                    normalized_value = (current_value - window.min()) / (window.max() - window.min())
                    if (window.min() == current_value):
                        normalized_value = 1
                    normalized_values.append(normalized_value)

            # Add normalized feature to the group DataFrame
            group_normalized[f'{feature}_normalized'] = normalized_values

        # Remove the first (window_size-1) rows for this ticker
        group_normalized = group_normalized.iloc[window_size-1:]

        # Append to the list of normalized DataFrames
        normalized_dfs.append(group_normalized)

    # Concatenate the normalized DataFrames
    normalized_df = pd.concat(normalized_dfs, ignore_index=True)

    return normalized_df
