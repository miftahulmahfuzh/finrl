# this training code is taken from:
# https://github.com/AI4Finance-Foundation/FinRL-Tutorials/blob/master/1-Introduction/FinRL_Ensemble_StockTrading_ICAIF_2020.ipynb
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import datetime

# %matplotlib inline
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent, DRLEnsembleAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

# from pprint import pprint

import sys
sys.path.append("../FinRL-Library")

import itertools

import os
from finrl.main import check_and_make_directories
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    # TRAIN_START_DATE,
    # TRAIN_END_DATE,
    # TEST_START_DATE,
    # TEST_END_DATE,
    # TRADE_START_DATE,
    # TRADE_END_DATE,
)
INDICATORS = ['macd',
               'rsi_30',
               'cci_30',
               'dx_30']
check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])

# from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
# from finrl.config_tickers import DOW_30_TICKER

# TRAIN_START_DATE = '2020-12-30'
# TRAIN_END_DATE = '2021-09-31'

# TEST_START_DATE = '2021-10-01'
# TEST_END_DATE = '2021-12-30'

# ydf = YahooDownloader(start_date = TRAIN_START_DATE,
#                      end_date = TEST_END_DATE,
#                      ticker_list = DOW_30_TICKER).fetch_data()

# print(ydf.head())  # Display the first 5 rows
# print(ydf.columns) # Display the column names
# from finrl.meta.preprocessor.preprocessors import FeatureEngineer
# yfe = FeatureEngineer(use_technical_indicator=True,
#                      tech_indicator_list = INDICATORS,
#                      use_turbulence=True,
#                      user_defined_feature = False)

# yprocessed = yfe.preprocess_data(ydf)
# yprocessed = yprocessed.copy()
# yprocessed = yprocessed.fillna(0)
# yprocessed = yprocessed.replace(np.inf,0)

# RAW DATASET
raw_df = pd.read_csv("/home/devmiftahul/trading_model/tuntun_data/Daily93Tuntun_2010-2021.csv")
raw_df = raw_df[pd.to_datetime(raw_df['date']) > pd.Timestamp('2009-01-01')]
raw_df = raw_df[~raw_df['tic'].str.contains('-')]
raw_df = raw_df.drop_duplicates(subset=['date', 'tic'])
print(raw_df)

# tic_counts = df['tic'].value_counts()
# sorted_tic_counts = tic_counts.sort_values(ascending=False)

# # Print the top 10 `tic` with their counts
# tickers = sorted_tic_counts.tail(30).index.tolist()

# Step 3: Count rows where 'volume' is 0 for each 'tic'
volume_zero_count = raw_df[raw_df['volume'] == 0].groupby('tic').size()

# Step 4: Sort the counts in descending order
sorted_volume_zero_count = volume_zero_count.sort_values(ascending=True)

# Step 5: Get the top n 'tic' with their counts
top_n_tic = sorted_volume_zero_count.head(30)
tickers = top_n_tic.index.tolist()
print(f"TICKERS: {tickers}")

df = raw_df[raw_df["tic"].isin(tickers)]
# print(df)

# PREPROCESS DATA
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
fe = FeatureEngineer(use_technical_indicator=True,
                     tech_indicator_list = INDICATORS,
                     use_turbulence=True,
                     user_defined_feature = False)

processed = fe.preprocess_data(df)
processed = processed.copy()
processed = processed.fillna(0)
processed = processed.replace(np.inf,0)
print(f"TOTAL TIC AFTER PROCESSED: {len(processed.tic.unique())}")
# print(processed)

stock_dimension = len(processed.tic.unique())
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "state_space": state_space,
    # "state_space": 176,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
    "print_verbosity":5
}

TRAIN_START_DATE = '2020-12-30'
TRAIN_END_DATE = '2021-09-31'

TEST_START_DATE = '2021-10-01'
TEST_END_DATE = '2021-12-30'
rebalance_window = 6 # rebalance_window is the number of days to retrain the model
validation_window = 6 # validation_window is the number of days to do validation and trading (e.g. if validation_window=63, then both validation and trading period will be 63 days)

import model_kwargs as m
ensemble_agent = DRLEnsembleAgent(df=processed,
                 train_period=(TRAIN_START_DATE,TRAIN_END_DATE),
                 val_test_period=(TEST_START_DATE,TEST_END_DATE),
                 rebalance_window=rebalance_window,
                 validation_window=validation_window,
                 **env_kwargs)

df_summary = ensemble_agent.run_ensemble_strategy(
    m.A2C_model_kwargs,
    m.PPO_model_kwargs,
    m.DDPG_model_kwargs,
    m.SAC_model_kwargs,
    m.TD3_model_kwargs,
    m.timesteps_dict
)
print(df_summary)

# EVALUATION
unique_trade_date = processed[(processed.date > TEST_START_DATE) & (processed.date <= TEST_END_DATE)].date.unique()
df_trade_date = pd.DataFrame({'datadate':unique_trade_date})

df_account_value=pd.DataFrame()
for i in range(rebalance_window+validation_window, len(unique_trade_date)+1,rebalance_window):
    temp = pd.read_csv('results/account_value_trade_{}_{}.csv'.format('ensemble',i))
    df_account_value = pd.concat([df_account_value, temp], ignore_index=True)
sharpe=(252**0.5)*df_account_value.account_value.pct_change(1).mean()/df_account_value.account_value.pct_change(1).std()
print('Sharpe Ratio: ',sharpe)
df_account_value=df_account_value.join(df_trade_date[validation_window:].reset_index(drop=True))
print(f"DATA TEST ACCOUNT VALUE:")
print(df_account_value)

print("\n==============Get Backtest Results===========")
now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

perf_stats_all = backtest_stats(account_value=df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)

