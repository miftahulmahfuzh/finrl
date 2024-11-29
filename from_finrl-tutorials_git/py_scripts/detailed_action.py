import pandas as pd
import numpy as np

def create_detailed_actions_excel(tickers, history_action, history_amount):
    """
    Create an Excel file with detailed trading actions for each ticker and day.

    Parameters:
    - tickers: List of ticker symbols
    - history_action: Dictionary of actions for each day
    - history_amount: Dictionary of amount of money at the end of each market close

    Returns:
    - DataFrame with detailed actions
    """
    # Prepare the columns
    columns = ['day'] + list(tickers) + ['amount_of_money_at_the_end_of_market_close']

    # Create a list to store data for each day
    data_rows = []

    # Iterate through days in the history_action
    for day, day_actions in history_action.items():
        # Create a row for this day
        row_data = [day]

        # Ensure day_actions has exactly 100 entries (one for each ticker)
        if len(day_actions) != len(tickers):
            print(f"Warning: Day {day} has {len(day_actions)} actions instead of {len(tickers)}")
            # Pad with default values if needed
            day_actions = day_actions + [(0, 0, 0)] * (len(tickers) - len(day_actions))

        # Add ticker actions to the row
        row_data.extend(day_actions)

        # Add amount of money at market close
        row_data.append(history_amount.get(day, np.nan))

        # Append the row to data
        data_rows.append(row_data)

    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=columns)

    # Save to Excel
    df.to_excel('detailed_actions.xlsx', sheet_name='detailed_actions', index=False)

    print("Excel file 'detailed_actions.xlsx' has been created successfully.")

    return df

# # Call the function with the provided variables
# # Assuming tickers, history_action, and history_amount are already defined
# result_df = create_detailed_actions_excel(tickers, history_action, history_amount)

# # Optional: Display first few rows to verify
# print(result_df.head())
