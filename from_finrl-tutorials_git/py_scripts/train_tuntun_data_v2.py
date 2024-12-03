# this script use portfolio optimization (EIIE) on andrew's data

import pandas as pd

raw_df = pd.read_csv("/home/devmiftahul/trading_model/tuntun_data/Daily93Tuntun_2010-2021.csv")
# raw_df = pd.read_csv("../tuntun_data/Daily93Tuntun_2010-2021.csv")
raw_df = raw_df[pd.to_datetime(raw_df['date']) > pd.Timestamp('2009-01-01')]
raw_df = raw_df[~raw_df['tic'].str.contains('-')]
raw_df = raw_df.drop_duplicates(subset=['date', 'tic'])

raw_df['date'] = pd.to_datetime(raw_df['date'])
raw_df = raw_df.sort_values('date').reset_index(drop=True)
date_mapping = {date: i+1 for i, date in enumerate(raw_df['date'].unique())}
raw_df['day'] = raw_df['date'].map(date_mapping)

# Step 1: Count rows where 'volume' is 0 for each 'tic'
volume_zero_count = raw_df[raw_df['volume'] == 0].groupby('tic').size()

# Step 2: Sort the counts in descending order
sorted_volume_zero_count = volume_zero_count.sort_values(ascending=True)

# Step 3: Get 100 tickers with the least amount of days where its 'volume' is 0
top_n_tic = sorted_volume_zero_count.head(100)
tickers = top_n_tic.index.tolist()
df = raw_df[raw_df["tic"].isin(tickers)]

df['date'] = pd.to_datetime(df['date'])

# Sort the DataFrame by 'date' to ensure chronological order
df = df.sort_values('date').reset_index(drop=True)

# Create a mapping of unique dates to sequential numbers starting from 1
date_mapping = {date: i+1 for i, date in enumerate(df['date'].unique())}

# Map the 'date' column to the new 'day' values
df['day'] = df['date'].map(date_mapping)

# Display the first few rows to verify
print(f"ADDED DAY COLUMN:")
print(df)

from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv
df_portfolio_train = df[df["day"] < 442]
df_portfolio_test = df[df["day"] >= 442]
TIME_WINDOW = 6

environment_train = PortfolioOptimizationEnv(
        df_portfolio_train,
        initial_amount=100000,
        comission_fee_pct=0.0025,
        time_window=TIME_WINDOW,
        features=["close", "high", "low"],
        time_column="day",
        normalize_df=None, # dataframe is already normalized
        tics_in_portfolio=tickers
    )
environment_test = PortfolioOptimizationEnv(
        df_portfolio_test,
        initial_amount=100000,
        comission_fee_pct=0.0025,
        time_window=TIME_WINDOW,
        features=["close", "high", "low"],
        time_column="day",
        normalize_df=None, # dataframe is already normalized
        tics_in_portfolio=tickers
    )

import torch
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

from finrl.agents.portfolio_optimization.architectures import EIIE
from finrl.agents.portfolio_optimization.models import DRLAgent
model_kwargs = {
    "lr": 0.001,
    "policy": EIIE,
}
policy_kwargs = {
    "initial_features": 3,
    "k_size": 3,
    "conv_mid_features": 2,
    "conv_final_features": 20,
    "time_window": 6
}

model = DRLAgent(environment_train).get_model("pg", device, model_kwargs, policy_kwargs)
DRLAgent.train_model(model, episodes=10)

import os
d = "trained_models/eiie"
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

import matplotlib.pyplot as plt
# %matplotlib inline

plt.plot(UBAH_results["test"], label="Buy and Hold")
plt.plot(EIIE_results["test"], label="EIIE")

plt.xlabel("Days")
plt.ylabel("Portfolio Value")
plt.title("Performance in testing period")
plt.legend()

plt.show()
