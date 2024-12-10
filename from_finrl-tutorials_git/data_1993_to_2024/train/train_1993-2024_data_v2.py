import pandas as pd
from datetime import datetime
import os
from utils import (
    split_data_based_on_date,
)

fprocessed = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/combined_data/923_tuntun_api_data_with_features.csv"
processed = pd.read_csv(fprocessed)
tickers = processed["tic"].unique()
tickers = tickers[:30]

processed = processed[processed["tic"].isin(tickers)]
print(f"TICKERS IN DATA: {processed['tic'].unique()}")
unique_days = processed["day"].unique()
print(f"TOTAL UNIQUE DAYS: {len(unique_days)}")

TRAIN_START_DATE = "1993-01-04"
TRAIN_END_DATE = "2022-12-31"
TEST_START_DATE = "2023-01-01"
TEST_END_DATE = "2024-11-26"
train_processed, test_processed = split_data_based_on_date(processed, TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE)

days = processed["day"].unique()
print(f"total all days in dataset {len(days)}")
days = train_processed["day"].unique()
print(f"total all days in train data {len(days)}")
days = test_processed["day"].unique()
print(f"total all days in test data {len(days)}")

df_portfolio_train = train_processed
df_portfolio_test = test_processed

from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv

TIME_WINDOW = 50
features=["open", "close", "high", "low"]
initial_features = len(features)
initial_amount = 1e8
comission_fee_pct = 0.0025
detailed_actions_file = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/train/v2_result/eiie/detailed_actions_eiie.csv"

environment_train = PortfolioOptimizationEnv(
        df_portfolio_train,
        initial_amount=initial_amount,
        comission_fee_pct=comission_fee_pct,
        time_window=TIME_WINDOW,
        features=features,
        time_column="date",
        normalize_df=None, # dataframe is already normalized
        tics_in_portfolio="all",
        is_train=True,
    )
environment_test = PortfolioOptimizationEnv(
        df_portfolio_test,
        initial_amount=initial_amount,
        comission_fee_pct=comission_fee_pct,
        time_window=TIME_WINDOW,
        features=features,
        time_column="date",
        normalize_df=None, # dataframe is already normalized
        tics_in_portfolio="all",
        is_train=False,
        detailed_actions_file=detailed_actions_file,
    )

# df_portfolio_test
print(f"TOTAL MARKET DAYS FOR TEST: {len(df_portfolio_test['day'].unique())}")

from finrl.agents.portfolio_optimization.architectures import EIIE
from finrl.agents.portfolio_optimization.models import DRLAgent
import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

MODE = "test"

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
    DRLAgent.train_model(model, episodes=40)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# model_name = f"eiie_{timestamp}"
model_name = "eiie"
d = f"v2_result/{model_name}"
os.makedirs(d, exist_ok=True)
model_path = f"{d}/policy_EIIE.pt"

if MODE == "train":
    torch.save(model.train_policy.state_dict(), model_path)

EIIE_results = {
    "train": environment_train._asset_memory["final"],
    "test": {},
}

# instantiate an architecture with the same arguments used in training
# and load with load_state_dict.
policy = None
if MODE == "train":
    policy = model.train_policy
else:
    policy = EIIE(**policy_kwargs)
    policy.load_state_dict(torch.load(model_path))

# testing
DRLAgent.DRL_validation(model, environment_test, policy=policy)
EIIE_results["test"] = environment_test._asset_memory["final"]
