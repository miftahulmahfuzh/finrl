import pandas as pd
from utils import (
    process_ohlcv,
)

def count_unusable_close_row_per_tic(df):
    # Group by ticker and count rows where close price is 1
    unusable_rows = df[df['close'] == 1].groupby('tic').size().reset_index(name='unusable_rows')

    # Sort in descending order of unusable rows
    unusable_rows = unusable_rows.sort_values('unusable_rows', ascending=False)

    return unusable_rows

# fprocessed = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/combined_data/923_tuntun_api_data_with_features.csv"
# processed = pd.read_csv(fprocessed)
# print(len(processed))

tickers = open("tickers_v2.txt").read().splitlines()
# df = processed[processed["tic"].isin(tickers)]
# df = process_ohlcv(df)
fprocessed_v2 = f"/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/combined_data/{len(tickers)}_tuntun_api_data_with_features.csv"
# df.to_csv(fprocessed_v2, index=False)
df = pd.read_csv(fprocessed_v2)
print(len(df))
# stat = count_unusable_close_row_per_tic(df)
# print(stat)

df['dfdate'] = pd.to_datetime(df['date'])
TRAIN_START_DATE = "2010-11-29"
mask = (df['dfdate'] >= TRAIN_START_DATE)
df = df[mask].copy()
fprocessed_v3 = f"/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/combined_data/{len(tickers)}_tuntun_api_data_with_features_v3.csv"
df = df.drop(columns=['dfdate'])
df.to_csv(fprocessed_v3, index=False)
