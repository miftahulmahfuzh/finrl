import pandas as pd
from utils import (
    split_data_based_on_date,
)

# fprocessed = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/processed_by_gavin/preprocessed_data_with_features.csv"
td = "combined_data"
fprocessed = f"/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/{td}/923_tuntun_api_data_with_features.csv"

processed = pd.read_csv(fprocessed)
print(processed)

from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.config import INDICATORS
print(INDICATORS)

print(f"TOTAL DAYS: {len(processed['day'].unique())}")

TRAIN_START_DATE = '1993-01-04'
TRAIN_END_DATE = '2022-12-31'
TEST_START_DATE = '2023-01-01'
TEST_END_DATE = '2024-12-06'
train_processed, test_processed = split_data_based_on_date(processed, TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE)

traded_tickers = ["ASII"]
tmp_df = test_processed[test_processed["tic"].isin(traded_tickers)]
print(f"DF tic 'ASII': {tmp_df}")

days = processed["day"].unique()
print(f"total all days in dataset {len(days)}")
days = train_processed["day"].unique()
print(f"total all days in train data {len(days)}")
days = test_processed["day"].unique()
print(f"total all days in test data {len(days)}")

# TRAIN
price_array, tech_array, turbulence_array, train_tickers, train_dates = df_to_array(train_processed, INDICATORS)

env_config = {
    "price_array": price_array,
    "tech_array": tech_array,
    "turbulence_array": turbulence_array,
    "if_train": True,
}
env_instance = StockTradingEnv(config=env_config)
ERL_PARAMS = {"learning_rate": 3e-6,"batch_size": 2048,"gamma":  0.985,
        "seed":312,"net_dimension":[256,128], "target_step":5000, "eval_gap":30,
        "eval_times":1}
MAIN_DIR = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/train/model_v3"

from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = "ppo"
cwd = f"{MAIN_DIR}/{model_name}_{timestamp}"

print(f"TOTAL TICKERS: {len(tickers)}")

DRLAgent_erl = DRLAgent
break_step = 1e5
agent = DRLAgent(
    env=StockTradingEnv,
    price_array=price_array,
    tech_array=tech_array,
    turbulence_array=turbulence_array,
)
model = agent.get_model(model_name, model_kwargs=ERL_PARAMS)
trained_model = agent.train_model(
    model=model, cwd=cwd, total_timesteps=break_step
)

# TEST
test_price_array, test_tech_array, test_turbulence_array, test_tickers, test_dates = df_to_array(test_processed, INDICATORS)
env_config = {
    "price_array": test_price_array,
    "tech_array": test_tech_array,
    "turbulence_array": test_turbulence_array,
    "if_train": False,
}
env_instance = StockTradingEnv(config=env_config)

# load elegantrl needs state dim, action dim and net dim
net_dimension = [256, 128] # ERL_PARAMS.get("net_dimension", 2**7)
print(f"price_array: {len(test_price_array)} days")

DRLAgent_erl = DRLAgent
episode_total_assets = DRLAgent_erl.DRL_prediction(
    model_name=model_name,
    cwd=cwd,
    net_dimension=net_dimension,
    environment=env_instance,
)
print(f"EPISODE TOTAL ASSETS: {episode_total_assets}")

# SAVE TEST RESULT
history_action = env_instance.history_action
history_amount = env_instance.history_amount
history_total = env_instance.history_total
turbulence_bool = env_instance.turbulence_bool
ccwd = cwd.split("/")[-1]
excel_path = f"{MAIN_DIR}/detailed_actions_{ccwd}.xlsx"
create_detailed_actions_excel(
    excel_path, tickers, history_action,
    history_amount, test_turbulence_array,
    turbulence_bool, test_dates)
