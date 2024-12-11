import pandas as pd
from utils import (
    get_quality_tickers,
    _temporal_variation_df
)
from tqdm import tqdm

def get_tickers_with_extreme_price_increase(df, date):
    # filter df to 3 dates: 1 market open day before date, date, and 1 market open day after date
    # check the price difference between 1-date-prev and date
    # tickers_prev: print all tic where close price on 1-date-prev is 1 and close price on date is > 1
    # check the price difference between date and 1-date-after
    # tickers_after: print all tic where close price on close price on date is 1 and close price on 1-date-after is > 1

    # Convert the input date to datetime if it's not already
    target_date = pd.to_datetime(date)

    # Sort the dataframe by date and ticker to ensure correct ordering
    df_sorted = df.sort_values(['dfdate', 'tic'])

    # Get unique dates and ensure they are sorted
    unique_dates = sorted(df_sorted['dfdate'].unique())

    # Find the indices of the target date in the unique dates list
    try:
        target_date_index = unique_dates.index(target_date)
    except ValueError:
        print(f"Target date {target_date} not found in the dataset")
        return [], []

    # Ensure we have dates before and after the target date
    if target_date_index == 0 or target_date_index == len(unique_dates) - 1:
        print("Not enough surrounding dates for analysis")
        return [], []

    # Get the previous and next dates
    prev_date = unique_dates[target_date_index - 1]
    next_date = unique_dates[target_date_index + 1]

    # Filter dataframes for each date
    df_prev = df_sorted[df_sorted['dfdate'] == prev_date]
    df_target = df_sorted[df_sorted['dfdate'] == target_date]
    df_next = df_sorted[df_sorted['dfdate'] == next_date]
    tmp_df = pd.concat([df_prev, df_target, df_next], reset_index)

    # Find tickers with extreme price increases
    # Tickers that go from 1 to > 1 from previous date to target date
    tickers_prev = df_prev[
        (df_prev['close'] == 1) &
        (df_target.set_index('tic')['close'] > 1)
    ]['tic'].tolist()

    # Tickers that go from 1 to > 1 from target date to next date
    tickers_after = df_target[
        (df_target['close'] == 1) &
        (df_next.set_index('tic')['close'] > 1)
    ]['tic'].tolist()

    return tickers_prev, tickers_after, tmp_df
# check BBCA september 30th 2005

def get_first_trading_date(df, tic):
    # TODO: return the earliest date this ticker has close price > 1
    df = df[df["tic"] == tic]
    unique_dates = sorted(df['dfdate'].unique())
    df_tic_sorted = df.sort_values('dfdate')

    # Find the first date where close price is greater than 1
    first_trading_day = df_tic_sorted[df_tic_sorted['close'] > 1]['dfdate'].min()

    return first_trading_day

fprocessed = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/combined_data/923_tuntun_api_data_with_features.csv"
processed = pd.read_csv(fprocessed)
df = get_quality_tickers(processed)
df['dfdate'] = pd.to_datetime(df['date'])
df = df[["date", "dfdate", "tic", "high", "low", "close"]]
# df = df[df["tic"] == "BBCA"]
df['dfdate'] = pd.to_datetime(df['date'])

# TRAIN_START_DATE = "1993-10-01"
# TRAIN_END_DATE = "1993-10-05"
# TRAIN_START_DATE = "2000-05-31"
# TRAIN_END_DATE = "2000-06-05"
# TRAIN_START_DATE = "2005-09-30"
# TRAIN_END_DATE = "2005-10-05"

# # Create boolean masks for training and testing data
# train_mask = (df['dfdate'] >= TRAIN_START_DATE) & (df['dfdate'] <= TRAIN_END_DATE)
# train_df = df[train_mask].copy()
# print(train_df)

# tic = "BBCA"
# init_date = get_first_trading_date(df, tic)
# print(f"First trading day for {tic} in the data is: {init_date}")

# DATE = "1993-10-04"
# dates = open("problematic_dates.txt").read().splitlines()
# comdf = None
# for date in dates[:1]:
#     tickers_prev, tickers_after, tmp_df = get_tickers_with_extreme_price_increase(df, date)
#     print(f"\n{date}")
#     print(f"Problematic Tickers 1 Day Previous: {tickers_prev}")
#     print(f"Problematic Tickers 1 Day After: {tickers_after}")
    # if comdf is None:
    #     comdf = tmp_df
    # else:

tickers = sorted(df["tic"].unique())
list_tic = []
for tic in tqdm(tickers, total=len(tickers)):
    list_tic.append({"Tic": tic, "IPO": get_first_trading_date(df, tic)})

lt_df = pd.DataFrame(list_tic)
lt_df = lt_df.sort_values(by='IPO', ascending=False)
fname = "xlsx/ipo.xlsx"
lt_df.to_excel(fname, sheet_name="ipo", index=False)
print(f"List IPO is saved to {fname}")
