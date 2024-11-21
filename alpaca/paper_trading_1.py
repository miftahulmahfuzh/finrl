# taken from: https://github.com/AI4Finance-Foundation/FinRL-Tutorials/blob/master/3-Practical/FinRL_PaperTrading_Demo.ipynb
from __future__ import annotations

from finrl.config_tickers import DOW_30_TICKER
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.meta.env_stock_trading.env_stock_papertrading import AlpacaPaperTrading
from finrl.meta.data_processor import DataProcessor
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

import numpy as np
import pandas as pd
import numpy.random as rd
from copy import deepcopy

from actor_ppo import (
    ActorPPO,
    CriticPPO,
    build_mlp,
    Config,
    get_gym_env_args,
    kwargs_filter,
    build_env,
    AgentBase,
    AgentPPO,
    PendulumEnv,
    train_agent,
    render_agent,
    Evaluator,
    get_rewards_and_steps,
)

from drl_agent import (
    MODELS,
    OFF_POLICY_MODELS,
    ON_POLICY_MODELS,
    DRLAgent,
)

from train_test import (train, test)

from finrl.config import ERL_PARAMS
from finrl.config import RLlib_PARAMS
from finrl.config import SAC_PARAMS
from finrl.config import TRAIN_END_DATE
from finrl.config import TRAIN_START_DATE

ticker_list = DOW_30_TICKER
print(f"TICKER LIST: {ticker_list}")
action_dim = len(DOW_30_TICKER)
print(f"INDICATORS: {INDICATORS}")
state_dim = 1 + 2 + 3 * action_dim + len(INDICATORS) * action_dim
print(f"STATE DIM: {state_dim}")

API_KEY = "PKSCD25SIIWWGDU6MA2S"
API_SECRET = "cm90rIwfwi6kggGJiV7aeviac05wuKeKCLijifEt"
API_BASE_URL = 'https://paper-api.alpaca.markets'
data_url = 'wss://data.alpaca.markets'
env = StockTradingEnv

DP = DataProcessor(
    data_source = 'alpaca',
    API_KEY = API_KEY,
    API_SECRET = API_SECRET,
    API_BASE_URL = API_BASE_URL
)

data = DP.download_data(
    start_date = '2021-10-04',
    end_date = '2021-10-08',
    ticker_list = ticker_list,
    time_interval= '1Min'
)

print(f"TOTAL TIMESTAMP: {data['timestamp'].nunique()}")
print("\nDATA")
print(data)

data = DP.clean_data(data)
data = DP.add_technical_indicator(data, INDICATORS)
data = DP.add_vix(data)

print(f"DATA SHAPE: {data.shape}")

price_array, tech_array, turbulence_array = DP.df_to_array(data, if_vix=True)

print(f"PRICE ARRAY: {price_array}")

ERL_PARAMS = {"learning_rate": 3e-6,"batch_size": 2048,"gamma":  0.985,
        "seed":312,"net_dimension":[128,64], "target_step":5000, "eval_gap":30,
        "eval_times":1}

env = StockTradingEnv
#if you want to use larger datasets (change to longer period), and it raises error,
#please try to increase "target_step". It should be larger than the episode steps.

train(start_date = '2022-08-25',
      end_date = '2022-09-02',
      ticker_list = ticker_list,
      data_source = 'alpaca',
      time_interval= '1Min',
      technical_indicator_list= INDICATORS,
      drl_lib='elegantrl',
      env=env,
      model_name='ppo',
      if_vix=True,
      API_KEY = API_KEY,
      API_SECRET = API_SECRET,
      API_BASE_URL = API_BASE_URL,
      erl_params=ERL_PARAMS,
      cwd='./papertrading_erl', #current_working_dir
      break_step=1e5
)

account_value_erl=test(
    start_date = '2022-09-01',
    end_date = '2022-09-02',
    ticker_list = ticker_list,
    data_source = 'alpaca',
    time_interval= '1Min',
    technical_indicator_list= INDICATORS,
    drl_lib='elegantrl',
    env=env,
    model_name='ppo',
    if_vix=True,
    API_KEY = API_KEY,
    API_SECRET = API_SECRET,
    API_BASE_URL = API_BASE_URL,
    cwd='./papertrading_erl',
    net_dimension = ERL_PARAMS['net_dimension']
)


