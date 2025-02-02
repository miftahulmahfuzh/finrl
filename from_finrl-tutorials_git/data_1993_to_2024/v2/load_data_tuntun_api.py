import pandas as pd
from glob import glob
from tqdm import tqdm
from utils import (
    clean_ohlcv,
    add_needed_dates,
    validate_tic_dates,
)

dates_file = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/v2/dates_tuntun_api_v2.txt"
needed_dates = open(dates_file).read().splitlines()
date2day = {date: day for day, date in enumerate(needed_dates)}

# d = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/tuntun_scripts/csv_2024-12-06/*.csv"
d = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/tuntun_scripts/csv_2024-12-23/*.csv"
fnames = glob(d)
print(f"Total csv in dir: {len(fnames)}")
total_in_each_tic = {}
list_to_redo_fetch = []
all_df = None
tickers = []
# dates = []
# max_dates = 0
for fname in tqdm(fnames, total=len(fnames)):
    dfi = pd.read_csv(fname)
    dfi = clean_ohlcv(dfi)
    tic = fname.split("/")[-1][:4]

    # dfi_dates = sorted(dfi["date"].unique())
    dfi = add_needed_dates(dfi, needed_dates, tic)
    # len_date = len(dfi_dates)
    # if len_date > max_dates:
    #     max_dates = len_date
    #     dates = dfi_dates

    if all_df is not None:
        all_df = pd.concat([all_df, dfi], ignore_index=True)
        tickers.append(tic)
    else:
        all_df = dfi
        tickers.append(tic)

# dates_str = "\n".join(dates)
# with open("dates_tuntun_api_v2.txt", "w+") as f:
#     f.write(dates_str)

new_total_for_each_tic, total_unique_dates = validate_tic_dates(all_df)
print("Total unique dates in the dataset:", total_unique_dates)

start_date = needed_dates[0]
end_date = needed_dates[-1]
d = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/data"
fname = f"{d}/{len(tickers)}_tickers_tuntun_api_v3_{start_date}_{end_date}.csv"
all_df.to_csv(fname, index=False)
print(f"2024 ohlcv data is saved to:\n{fname}")
