import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from utils import (
    validate_tic_dates,
    clean_ohlcv,
    calculate_turbulence,
)
from finrl.main import check_and_make_directories
from finrl.config import (
    # DATA_SAVE_DIR,
    # TRAINED_MODEL_DIR,
    # TENSORBOARD_LOG_DIR,
    # RESULTS_DIR,
    INDICATORS,
)
# check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])

# d = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/data"
# fname = f"{d}/937_tickers_tuntun_api_2024-03-13_2024-11-26.csv"
# all_df = pd.read_csv(fname)

# all_df = clean_ohlcv(all_df)
# new_total_for_each_tic, total_unique_dates = validate_tic_dates(all_df)
# print("Total unique dates in the dataset:", total_unique_dates)

d = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/data"
# fname = f"{d}/953_tickers_tuntun_api_v3_2024-03-13_2024-12-06.csv"
fname = f"{d}/75_tickers_tuntun_api_v3_2024-03-13_2024-12-06.csv"
df_api = pd.read_csv(fname)
# new_total_for_each_tic, total_unique_dates = validate_tic_dates(all_df)
# print("Total unique dates in the dataset:", total_unique_dates)
columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'tic']
tickers_api = sorted(df_api["tic"].unique())

# turbulence = calculate_turbulence(all_df)
# print(f"TURBULENCE: {turbulence}")

fname_adw = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/processed_by_gavin/preprocessed_data.csv"
df_adw = pd.read_csv(fname_adw)
df_adw = df_adw.rename(columns={"tick": "tic"})
df_adw = df_adw[df_adw["tic"].isin(tickers_api)]
df_adw = df_adw[columns]

all_df = pd.concat([df_adw, df_api], ignore_index=True)

dates = sorted(all_df["date"].unique())
dates2day = {date:day for day, date in enumerate(dates, start=1)}
all_df["day"] = all_df["date"].map(dates2day)

days = len(dates)
first_date = dates[0]
last_date = dates[-1]
print(f"Total days in dataset {days}")
print(f"First date: {first_date}")
print(f"Last date: {last_date}")

from finrl.meta.preprocessor.preprocessors import FeatureEngineer
fe = FeatureEngineer(use_technical_indicator=True,
                     tech_indicator_list=INDICATORS,
                     use_turbulence=True,
                     user_defined_feature=False)

processed = fe.preprocess_data(all_df)
ntic = len(processed["tic"].unique())
td = "combined_data"
fprocessed = f"/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/{td}/{ntic}_tic_with_features.csv"
processed.to_csv(fprocessed, index=False)
print(f"Feature Engineer result for ohlcv data 1993-2024 is saved to:\n{fprocessed}")
