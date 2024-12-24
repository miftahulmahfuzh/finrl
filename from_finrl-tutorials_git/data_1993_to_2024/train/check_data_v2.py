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

def replace_row(df, date, tic, data):
    # TODO: implement function to replace df row
    # get index of df with 'tic' == tic and 'date' == date
    # replace df[index] with data
    index = df[(df['date'] == date) & (df['tic'] == tic)].index

    # Check if the row exists
    if len(index) == 0:
        print(f"No row found for date {date} and ticker {tic}")
        return df

    # If multiple rows found, use the first one
    index = index[0]

    # Replace the row with the new data
    for column, value in data.items():
        df.at[index, column] = value

    return df

MODE = "test"
# fprocessed_v3 = f"/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/combined_data/75_tuntun_api_data_with_features_v3.csv"
# fprocessed_v3a = f"/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/combined_data/75_tuntun_api_data_with_features_v3-a.csv"
# fprocessed_v3a = f"/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/data/75_tickers_tuntun_api_v3_2024-03-13_2024-12-06.csv"
# fprocessed_v3a = f"/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/combined_data/75_tic_with_features.csv"
# fprocessed_v3a = f"/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/combined_data/75_tic_v3-a_test_12_features_n_by_previous_time.xlsx"
# fprocessed_v3a = f"/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/combined_data/75_tic_v3-a_{MODE}_12_features_n_by_previous_time.xlsx"
# fgavin = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/processed_by_gavin/preprocessed_data.csv"
# d = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/data"
# fprocessed_v3a = f"{d}/937_tickers_tuntun_api_2024-03-13_2024-11-26.csv"
fprocessed_v4a = f"/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/combined_data/75_tic_with_features_v4-a.csv"
# fprocessed_v4b = f"/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/combined_data/75_tic_with_features_v4-b.csv"
# fprocessed_v5 = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/combined_data/75_tic_with_features_v5.csv"

df = pd.read_csv(fprocessed_v4a)
# print(df.columns)
# df = pd.read_csv(fprocessed_v3a)
# df = pd.read_excel(fprocessed_v3a, sheet_name="test")

# skip_dates = ["2013-05-25", "2013-05-26", "2024-03-23"]
# df = df[~df["date"].isin(skip_dates)]
# # delete_columns = ["Unnamed: 0", "Unnamed: 1"]
# # df.drop(columns=delete_columns, inplace=True, errors="ignore")

# dates = sorted(df['date'].unique())
# date2day = {date:day for day, date in enumerate(dates, start=1)}
# df["day"] = df["date"].map(date2day)
# fprocessed_v4b = f"/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/combined_data/75_tic_with_features_v4-b.csv"
# df.to_csv(fprocessed_v4b, index=False)
# print(f"Non Market Days has been removed in df. data saved to:\n{fprocessed_v4b}")
# exit()

# tickers = sorted(df["tic"].unique())
# with open(f"{len(tickers)}_tickers.txt", "w+") as f:
#     x = "\n".join(tickers)
#     f.write(x)

# REPLACE 'BAYU' ROW
# data_dict = {
#     'date': '2017-12-04',
#     'tic': 'BAYU',
#     'close': 1250.0,
#     'high': 1250.0,
#     'low': 1250.0,
#     'open': 1250.0,
#     'volume': 30000.0,
#     'day': 6088,
#     'macd': -76.09103421529030,
#     'boll_ub': 1752.279572017420,
#     'boll_lb': 631.3454279825710,
#     'rsi_30': 51.04593526101920,
#     'cci_30': 34.38349592195740,
#     'dx_30': 4.859308888669270,
#     'close_30_sma': 1211.875,
#     'close_60_sma': 1225.8541666666600,
#     'turbulence': 321340116.78699200
# }
# df = replace_row(df, "2017-12-04", "BAYU", data_dict)
# fprocessed_v3a = f"/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/combined_data/75_tic_with_features_v4-a.csv"
# df.to_csv(fprocessed_v3a)
# exit()

df['dfdate'] = pd.to_datetime(df['date'])
# fprocessed = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/combined_data/923_tuntun_api_data_with_features.csv"
# processed = pd.read_csv(fprocessed)
# df = get_quality_tickers(processed)
# df = df[["date", "dfdate", "tic", "high", "low", "close", "volume"]]
# df = df[["date", "dfdate", "tic", "high", "low", "close", "volume", "macd"]]
df = df[df["tic"] == "BAYU"]
# df = df[df["tic"] == "TOTL"]
# df = df[df["tic"] == "TGKA"]
# df = df[df["tic"] == "ELSA"]

# data = {}

# TRAIN_START_DATE = "1993-10-01"
# TRAIN_END_DATE = "1993-10-05"
TRAIN_START_DATE = "2017-11-27"
TRAIN_END_DATE = "2017-12-07"
# TRAIN_START_DATE = "2013-05-20"
# TRAIN_END_DATE = "2013-05-30"


# Create boolean masks for training and testing data
train_mask = (df['dfdate'] >= TRAIN_START_DATE) & (df['dfdate'] <= TRAIN_END_DATE)
train_df = df[train_mask].copy()
train_df = train_df.drop(columns=["dfdate"])
train_df.to_csv(f"csv/custom_df_{MODE}.csv", index=False)
print(f"FILE:\n{fprocessed_v4a}\nDATE RANGE: {TRAIN_START_DATE} TO {TRAIN_END_DATE}")
print(train_df)
exit()
# CHECK NAN IN TRAIN / DEV / TEST
# dd = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/train/v2_result/eiie_20241220_095522_profitable_in_train_nan_in_dev_and_test"
# dev_path = f"{dd}/result_eiie_dev.xlsx"
# dev_df = pd.read_excel(dev_path, sheet_name="detailed_actions")

# print the first row["date"] where row["portfolio"] is nan
# nan_row = dev_df[dev_df['portfolio'].isna()].iloc[0]

# # Print the date of the first NaN row in 'portfolio'
# print("The first date where 'portfolio' is NaN:", nan_row["date"])

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

# Save IPO List
# tickers = sorted(df["tic"].unique())
# list_tic = []
# for tic in tqdm(tickers, total=len(tickers)):
#     list_tic.append({"Tic": tic, "IPO": get_first_trading_date(df, tic)})

# lt_df = pd.DataFrame(list_tic)
# lt_df = lt_df.sort_values(by='IPO', ascending=False)
# fname = "xlsx/ipo.xlsx"
# lt_df.to_excel(fname, sheet_name="ipo", index=False)
# print(f"List IPO is saved to {fname}")
