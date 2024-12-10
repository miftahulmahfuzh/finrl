import pandas as pd
from datetime import datetime
import os
from utils import (
    split_data_based_on_date,
    get_quality_tickers,
    # clean_ohlcv_data,
    process_ohlcv,
)

fprocessed = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/combined_data/923_tuntun_api_data_with_features.csv"
processed = pd.read_csv(fprocessed)
# tickers = processed["tic"].unique()
# tickers = tickers[:30]
processed = get_quality_tickers(processed)
# processed = clean_ohlcv_data(processed)

# save it once
# fprocessed_quality = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/combined_data/121_quality_tic_with_features.csv"
# processed.to_csv(fprocessed_quality, index=False)

tickers = sorted(processed["tic"].unique())
# print(tickers)
# print(len(tickers))
print(f"TICKERS IN DATA ({len(tickers)}): {processed['tic'].unique()}")

# unique_days = processed["day"].unique()
# print(f"TOTAL UNIQUE DAYS: {len(unique_days)}")

TRAIN_START_DATE = "1993-01-04"
TRAIN_END_DATE = "2022-12-31"
DEV_START_DATE = "2023-01-01"
DEV_END_DATE = "2023-12-31"
TEST_START_DATE = "2024-01-01"
TEST_END_DATE = "2024-11-26"
x = split_data_based_on_date(
        processed,
        TRAIN_START_DATE,
        TRAIN_END_DATE,
        DEV_START_DATE,
        DEV_END_DATE,
        TEST_START_DATE,
        TEST_END_DATE)
train_processed, dev_processed, test_processed = x
test_processed = process_ohlcv(test_processed)

print("\n====================================")
days = processed["day"].unique()
print(f"Total all days in dataset {len(days)}")
days = train_processed["day"].unique()
print(f"Total all days in train data {len(days)}")
days = dev_processed["day"].unique()
print(f"Total all days in dev data {len(days)}")
days = test_processed["day"].unique()
print(f"Total all days in test data {len(days)}")
print("====================================\n")

df_portfolio_train = train_processed
df_portfolio_dev = dev_processed
df_portfolio_test = test_processed

from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"eiie_{timestamp}"
main_dir = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/train"
d = f"{main_dir}/v2_result/{model_name}"
os.makedirs(d, exist_ok=True)

MODE = "train"
TIME_WINDOW = 50
features=["close", "high", "low"]
initial_features = len(features)
initial_amount = 1e9
comission_fee_pct = 0.0025

detailed_actions_file = f"{d}/result_eiie_{MODE}.xlsx"
environment_train = PortfolioOptimizationEnv(
        df_portfolio_train,
        initial_amount=initial_amount,
        comission_fee_pct=comission_fee_pct,
        time_window=TIME_WINDOW,
        features=features,
        time_column="date",
        normalize_df=None, # dataframe is already normalized
        tics_in_portfolio="all",
        mode="train",
        detailed_actions_file=detailed_actions_file,
    )

from finrl.agents.portfolio_optimization.architectures import EIIE
from finrl.agents.portfolio_optimization.models import DRLAgent
import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)


model_kwargs = {
    "lr": 0.01,
    "policy": EIIE,
}
policy_kwargs = {
    "initial_features": initial_features,
    "k_size": 3,
    "conv_mid_features": 2,
    "conv_final_features": 20,
    "time_window": TIME_WINDOW
}

model = DRLAgent(environment_train).get_model("pg", device, model_kwargs, policy_kwargs)
if MODE == "train":
    DRLAgent.train_model(model, episodes=1)

model_path = f"{d}/policy_EIIE.pt"

if MODE == "train":
    torch.save(model.train_policy.state_dict(), model_path)

EIIE_results = {
    "train": environment_train._asset_memory["final"],
    "dev": {},
    "test": {}
}

# instantiate an architecture with the same arguments used in training
# and load with load_state_dict.
policy = None
if MODE == "train":
    policy = model.train_policy
else:
    policy = EIIE(**policy_kwargs)
    policy.load_state_dict(torch.load(model_path))

# dev
MODE = "dev"
detailed_actions_file = f"{d}/result_eiie_{MODE}.xlsx"
environment_dev = PortfolioOptimizationEnv(
        df_portfolio_dev,
        initial_amount=initial_amount,
        comission_fee_pct=comission_fee_pct,
        time_window=TIME_WINDOW,
        features=features,
        time_column="date",
        normalize_df=None, # dataframe is already normalized
        tics_in_portfolio="all",
        mode=MODE,
        detailed_actions_file=detailed_actions_file,
    )
DRLAgent.DRL_validation(model, environment_dev, policy=policy)
EIIE_results[MODE] = environment_dev._asset_memory["final"]

# testing
MODE = "test"
detailed_actions_file = f"{d}/result_eiie_{MODE}.xlsx"
environment_test = PortfolioOptimizationEnv(
        df_portfolio_test,
        initial_amount=initial_amount,
        comission_fee_pct=comission_fee_pct,
        time_window=TIME_WINDOW,
        features=features,
        time_column="date",
        normalize_df=None, # dataframe is already normalized
        tics_in_portfolio="all",
        mode=MODE,
        detailed_actions_file=detailed_actions_file,
    )
DRLAgent.DRL_validation(model, environment_test, policy=policy)
EIIE_results[MODE] = environment_test._asset_memory["final"]
