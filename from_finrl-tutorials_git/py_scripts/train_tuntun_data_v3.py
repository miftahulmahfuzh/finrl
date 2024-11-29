from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.meta.data_processor import DataProcessor
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from finrl.config import INDICATORS
from drl_agent import DRLAgent
from detailed_action_v2 import create_detailed_actions_excel

import pandas as pd
import numpy as np

def df_to_array(df, tech_indicator_list):
    df = df.copy()
    unique_ticker = df.tic.unique()
    if_first_time = True
    for tic in unique_ticker:
        if if_first_time:
            price_array = df[df.tic == tic][["close"]].values
            tech_array = df[df.tic == tic][tech_indicator_list].values
            turbulence_array = df[df.tic == tic]["turbulence"].values
            if_first_time = False
        else:
            price_array = np.hstack(
                [price_array, df[df.tic == tic][["close"]].values]
            )
            tech_array = np.hstack(
                [tech_array, df[df.tic == tic][tech_indicator_list].values]
            )
    tech_nan_positions = np.isnan(tech_array)
    tech_array[tech_nan_positions] = 0
    tech_inf_positions = np.isinf(tech_array)
    tech_array[tech_inf_positions] = 0
    return price_array, tech_array, turbulence_array, unique_ticker

d = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/tuntun_data"
processed_fout = f"{d}/processed_load_tuntun_data_v3.csv"
processed = pd.read_csv(processed_fout)

print(INDICATORS)
env = StockTradingEnv
drl_lib = "elegantrl"

train_processed = processed[processed["day"] < 442]
test_processed = processed[processed["day"] >= 442]
# price_array, tech_array, turbulence_array = df_to_array(train_processed, INDICATORS)

# env_config = {
#     "price_array": price_array,
#     "tech_array": tech_array,
#     "turbulence_array": turbulence_array,
#     "if_train": True,
# }
# env_instance = env(config=env_config)
ERL_PARAMS = {"learning_rate": 3e-6,"batch_size": 2048,"gamma":  0.985,
        "seed":312,"net_dimension":[128,64], "target_step":5, "eval_gap":30,
        "eval_times":1}

# read parameters
model_name = "ppo"
d = "/home/devmiftahul/trading_model/from_finrl-tutorials_git"
cwd = f"{d}/tuntun_papertrading_erl/{model_name}"
test_price_array, test_tech_array, test_turbulence_array, tickers = df_to_array(test_processed, INDICATORS)
env_config = {
    "price_array": test_price_array,
    "tech_array": test_tech_array,
    "turbulence_array": test_turbulence_array,
    "if_train": False,
}
env_instance = env(config=env_config)

# load elegantrl needs state dim, action dim and net dim
net_dimension = ERL_PARAMS.get("net_dimension", 2**7)
print(f"price_array: {len(test_price_array)} days")

if drl_lib == "elegantrl":
    DRLAgent_erl = DRLAgent
    episode_total_assets = DRLAgent_erl.DRL_prediction(
        model_name=model_name,
        cwd=cwd,
        net_dimension=net_dimension,
        environment=env_instance,
    )
    print(f"EPISODE TOTAL ASSETS: {episode_total_assets}")

history_action = env_instance.history_action
history_amount = env_instance.history_amount
turbulence_bool = env_instance.turbulence_bool
# print(len(tickers))
# print(tickers)
# print(history_action)
# print(history_amount)
create_detailed_actions_excel(tickers, history_action, history_amount, test_turbulence_array, turbulence_bool)
