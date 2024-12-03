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

features_csv = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/tuntun_scripts/processed_data/100_tickers_with_features.csv"
processed = pd.read_csv(features_csv)
df = processed.copy()
tickers = sorted(df.tic.unique())
print(f"TOTAL TICKERS: {len(tickers)}")

TRAIN_START_DATE = '2020-01-01'
TRAIN_END_DATE = '2022-12-31'
TEST_START_DATE = '2023-01-01'
TEST_END_DATE = '2023-12-31'
df_portfolio_train, df_portfolio_test = split_data_based_on_date(
        processed, TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE)

TIME_WINDOW = 63
features=["close", "high", "low"]
initial_features = len(features)

environment_train = PortfolioOptimizationEnv(
        df_portfolio_train,
        initial_amount=100000,
        comission_fee_pct=0.0025,
        time_window=TIME_WINDOW,
        features=features,
        time_column="day",
        normalize_df=None, # dataframe is already normalized
        tics_in_portfolio=tickers
    )
environment_test = PortfolioOptimizationEnv(
        df_portfolio_test,
        initial_amount=100000,
        comission_fee_pct=0.0025,
        time_window=TIME_WINDOW,
        features=features,
        time_column="day",
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

model = DRLAgent(environment_train).get_model("pg", device, model_kwargs, policy_kwargs)
print("Begin Training..")
# DRLAgent.train_model(model, episodes=1)

d = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/tuntun_api_trained_models/eiie"
os.makedirs(d, exist_ok=True)
model_path = f"{d}/policy_EIIE.pt"
torch.save(model.train_policy.state_dict(), model_path)

EIIE_results = {
    "train": environment_train._asset_memory["final"],
    "test": {},
}

# instantiate an architecture with the same arguments used in training
# and load with load_state_dict.
policy = EIIE(**policy_kwargs)
policy.load_state_dict(torch.load(model_path))

# testing
DRLAgent.DRL_validation(model, environment_test, policy=policy)
EIIE_results["test"] = environment_test._asset_memory["final"]
