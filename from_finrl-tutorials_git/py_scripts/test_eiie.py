import os
import torch
from finrl.agents.portfolio_optimization.architectures import EIIE
from finrl.agents.portfolio_optimization.models import DRLAgent
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv
from utils import split_data_based_on_date
import pandas as pd

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
# DONT USE THIS PATH BECAUSE ALL VOLUME HERE IS 0 (bug in fetch_data_using_api.py)
# processed_data_fname = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/tuntun_scripts/processed_data/processed_data_135_tickers.csv"

local = "/mnt/c/Users/mahfu/Downloads/tuntun/tuntun_ubuntu/github_me/finrl/from_finrl-tutorials_git"
features_csv = f"{local}/tuntun_scripts/processed_data/100_tickers_with_features.csv"
processed = pd.read_csv(features_csv)
tickers = sorted(processed.tic.unique())
print(f"TOTAL TICKERS: {len(tickers)}")
tickers = tickers[:10]
print(tickers)
processed = processed[processed["tic"].isin(tickers)]

TRAIN_START_DATE = '2020-01-01'
TRAIN_END_DATE = '2022-12-31'
TEST_START_DATE = '2023-01-01'
TEST_END_DATE = '2023-12-31'
df_portfolio_train, df_portfolio_test = split_data_based_on_date(
        processed, TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE)

TIME_WINDOW = 50
features=["close", "high", "low", "open"]
initial_features = len(features)

environment_train = PortfolioOptimizationEnv(
        df_portfolio_train,
        initial_amount=100000000,
        comission_fee_pct=0.0025,
        time_window=TIME_WINDOW,
        features=features,
        time_column="date",
        normalize_df=None, # dataframe is already normalized
        tics_in_portfolio=tickers
    )
environment_test = PortfolioOptimizationEnv(
        df_portfolio_test,
        initial_amount=100000000,
        comission_fee_pct=0.0025,
        time_window=TIME_WINDOW,
        features=features,
        time_column="date",
        normalize_df=None, # dataframe is already normalized
        tics_in_portfolio=tickers
    )

model_kwargs = {
    "lr": 0.0001,
    "policy": EIIE,
}
policy_kwargs = {
    "initial_features": initial_features,
    "k_size": 3,
    "conv_mid_features": 20,
    "conv_final_features": 200,
    "time_window": TIME_WINDOW
}

# MODE = "train"
MODE = "test"

model = DRLAgent(environment_train).get_model("pg", device, model_kwargs, policy_kwargs)
if MODE == "train":
    print("Begin Training..")
    DRLAgent.train_model(model, episodes=10)

d = f"{local}/tuntun_api_trained_models/eiie_10_episodes"
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
policy = EIIE(**policy_kwargs)
policy.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
if MODE == "train":
    policy = model.train_policy

# testing
DRLAgent.DRL_validation(model, environment_test, policy=policy)
EIIE_results["test"] = environment_test._asset_memory["final"]

# detailed_actions_file = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/tuntun_api_trained_models/eiie/detailed_actions.csv"
# environment_test.save_detailed_actions(detailed_actions_file)
# environment_test.finalize_actions(detailed_actions_file)
# print(f"Detailed actions of the agent is saved to {detailed_actions_file}")
