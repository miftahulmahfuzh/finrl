# this script use portfolio optimization (EIIE) on tuntun api data (136 tickers)
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv
import pandas as pd
finrl_processed_data_fname = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/tuntun_scripts/processed_data/processed_data_136_tickers_with_features.csv"
processed = pd.read_csv(finrl_processed_data_fname)
tickers = sorted(processed.tic.unique())
df = processed.copy()
df_portfolio_train = df[(df["day"] > 3000) & (df["day"] <= 3500)]
df_portfolio_test = df[df["day"] > 3500]
TIME_WINDOW = 63
features=["close", "high", "low", "macd", "rsi_30", "cci_30", "dx_30", "turbulence"]
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

import torch
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# print(device)
from finrl.agents.portfolio_optimization.architectures import EIIE
from finrl.agents.portfolio_optimization.models import DRLAgent
model_kwargs = {
    "lr": 0.0001,
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
# DRLAgent.train_model(model, episodes=10)

# import os
# d = "trained_models/eiie"
# os.makedirs(d, exist_ok=True)
# model_path = f"{d}/policy_EIIE.pt"
# torch.save(model.train_policy.state_dict(), model_path)

EIIE_results = {
    "train": environment_train._asset_memory["final"],
    "test": {},
}

# instantiate an architecture with the same arguments used in training
# and load with load_state_dict.
import os
d = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/tuntun_api_trained_models/eiie"
# os.makedirs(d, exist_ok=True)
model_path = f"{d}/policy_EIIE.pt"
policy = EIIE(**policy_kwargs)
policy.load_state_dict(torch.load(model_path))

# testing
print("\nEIIE TEST")
DRLAgent.DRL_validation(model, environment_test, policy=policy)
EIIE_results["test"] = environment_test._asset_memory["final"]

UBAH_results = {
    "train": {},
    "test": {},
}

PORTFOLIO_SIZE = len(tickers)

# train period
# terminated = False
# environment_train.reset()
# while not terminated:
#     action = [0] + [1/PORTFOLIO_SIZE] * PORTFOLIO_SIZE
#     _, _, terminated, _ = environment_train.step(action)
# UBAH_results["train"] = environment_train._asset_memory["final"]

# test period
print("\nUBAH TEST")
terminated = False
environment_test.reset()
while not terminated:
    action = [0] + [1/PORTFOLIO_SIZE] * PORTFOLIO_SIZE
    _, _, terminated, _ = environment_test.step(action)
UBAH_results["test"] = environment_test._asset_memory["final"]

# import matplotlib.pyplot as plt
# # %matplotlib inline

# plt.plot(UBAH_results["test"], label="Buy and Hold")
# plt.plot(EIIE_results["test"], label="EIIE")

# plt.xlabel("Days")
# plt.ylabel("Portfolio Value")
# plt.title("Performance in testing period")
# plt.legend()

# plt.show()
