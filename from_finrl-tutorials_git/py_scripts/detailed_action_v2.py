import pandas as pd
import numpy as np

def create_detailed_actions_excel(tickers, history_action, history_amount, turbulence_array, turbulence_bool):
    """
    Create an Excel file with detailed trading actions for tickers bought at least once.

    Parameters:
    - tickers: NumPy array or list of ticker symbols
    - history_action: Dictionary of actions for each day
    - history_amount: Dictionary of amount of money at the end of each market close

    Returns:
    - DataFrame with detailed actions
    """
    # Convert tickers to a list if it's a NumPy array
    tickers_list = tickers.tolist() if hasattr(tickers, 'tolist') else list(tickers)

    # Find tickers that have been bought at least once
    bought_tickers = set()

    # Iterate through all days and actions to find bought tickers
    for day_actions in history_action.values():
        for idx, action_tuple in enumerate(day_actions):
            # Check if the stock was bought (positive number of stocks purchased)
            if len(action_tuple) >= 3 and action_tuple[1] > 0 and action_tuple[2] > 0:
                bought_tickers.add(tickers_list[idx])

    # Convert to sorted list for consistent ordering
    bought_tickers = sorted(list(bought_tickers))

    # Prepare the columns
    columns = ['day', 'turbulence', 'sell_all'] + list(bought_tickers) + ['funds_on_market_close']

    # Create a list to store data for each day
    data_rows = []

    # Iterate through days in the history_action
    for day, day_actions in history_action.items():

        # Add turbulence value on that day
        # turbulence = turbulence_array.get(day, np.nan)
        day_index = day
        turbulence = turbulence_array[day_index] if day_index < len(turbulence_array) else np.nan

        # Add turbulence bool on that day
        # sell_all = turbulence_bool.get(day, np.nan)
        sell_all = turbulence_bool[day_index] if day_index < len(turbulence_bool) else np.nan

        # Create a row for this day
        row_data = [day, turbulence, sell_all]

        # Add actions for bought tickers
        day_bought_actions = []
        for ticker in bought_tickers:
            # Find the index of the ticker in the original tickers list
            ticker_index = tickers_list.index(ticker)

            # Get the action for this specific ticker
            if ticker_index < len(day_actions):
                day_bought_actions.append(day_actions[ticker_index])
            else:
                # Pad with default value if not enough actions
                day_bought_actions.append((0, 0, 0))

        # Extend row with bought ticker actions
        row_data.extend(day_bought_actions)

        # Add amount of money at market close
        row_data.append(history_amount.get(day, np.nan))

        # Append the row to data
        data_rows.append(row_data)

    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=columns)

    # Save to Excel
    df.to_excel('detailed_actions.xlsx', sheet_name='detailed_actions', index=False)

    print(f"Excel file 'detailed_actions.xlsx' has been created successfully.")
    print(f"Number of tickers with at least one buy action: {len(bought_tickers)}")

    return df

# # Call the function with the provided variables
# # Assuming tickers, history_action, and history_amount are already defined
# result_df = create_detailed_actions_excel(tickers, history_action, history_amount)

# # Optional: Display first few rows to verify
# print(result_df.head())
