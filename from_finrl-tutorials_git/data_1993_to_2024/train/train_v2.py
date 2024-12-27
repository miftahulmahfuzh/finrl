import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from datetime import datetime
import pandas as pd
import json
from utils import (
    split_data_based_on_date,
)
from normalization_utils import (
    sliding_windows_normalization,
)

# preprocessed_v3a is v3 that BAYU 2017-12-04 is updated manually in check_data.py
# preprocessed_v4 is produced by sliding_windows_normalization
# in v3-a, ELSA on 2024-11-20 has wrong ohlcv value
# fprocessed_v3a = f"/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/combined_data/75_tuntun_api_data_with_features_v3-a.csv"
# we fixed that on v4-a
# fprocessed_v4a = f"/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/combined_data/75_tic_with_features_v4-a.csv"
# in v4-a, there is a non market days ["2013-05-25", "2013-05-26", "2024-03-23"]
# we fixed that in v4-b
# fprocessed_v4b = f"/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/combined_data/75_tic_with_features_v4-b.csv"
# We got StopIteration Error (https://docs.google.com/document/d/1hiKPXlE9SPAW4vm4C-p5n9XQnr3zFj6Nydhj_XSd72g/edit?usp=sharing) in v4-b
# so, we try to rerun FeatureEngineer on v4b. that is v5
fprocessed_v5 = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/combined_data/75_tic_with_features_v5.csv"
processed = pd.read_csv(fprocessed_v5)
# processed = sliding_windows_normalization(processed)

# columns = ["date", "tic", "high", "high_normalized", "low", "low_normalized", "close", "close_normalized", "volume", "volume_normalized"]
# processed = processed[columns]
# fprocessed_v4 = f"/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/combined_data/75_tuntun_api_data_with_features_v4.csv"
# processed.to_csv(fprocessed_v4, index=False)

# fprocessed_yukun = f"/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/combined_data/75_tuntun_api_data_with_features_yukun.csv"
# processed = pd.read_csv(fprocessed_yukun)
print(processed)

tickers = sorted(processed["tic"].unique())
# tickers = tickers[:10]
# processed = processed[processed["tic"].isin(tickers)]
print(f"TICKERS IN DATA ({len(tickers)}): {processed['tic'].unique()}")

TRAIN_START_DATE = "2010-11-29"
TRAIN_END_DATE = "2022-12-31"
DEV_START_DATE = "2023-01-01"
DEV_END_DATE = "2023-12-31"
TEST_START_DATE = "2024-01-01"
TEST_END_DATE = "2024-11-26"

# TRAIN_START_DATE = "2023-11-29"
# TRAIN_END_DATE = "2024-04-18"
# DEV_START_DATE = "2024-04-19"
# DEV_END_DATE = "2024-06-18"
# TEST_START_DATE = "2024-06-19"
# TEST_END_DATE = "2024-08-18"
TIME_WINDOW = 50
x = split_data_based_on_date(
        processed,
        TRAIN_START_DATE,
        TRAIN_END_DATE,
        DEV_START_DATE,
        DEV_END_DATE,
        TEST_START_DATE,
        TEST_END_DATE)
train_processed, dev_processed, test_processed = x

train_dates = sorted(train_processed["date"].unique())[TIME_WINDOW:]
train_processed = train_processed[train_processed["date"].isin(train_dates)]

print("\n====================================")
days = processed["date"].unique()
print(f"Total days in dataset {len(days)}")
days = train_processed["date"].unique()
print(f"Total days in train data {len(days)}")
days = dev_processed["date"].unique()
print(f"Total days in dev data {len(days)}")
days = test_processed["date"].unique()
print(f"Total days in test data {len(days)}")
print("====================================\n")

df_portfolio_train = train_processed
df_portfolio_dev = dev_processed
df_portfolio_test = test_processed

# df_test = df_portfolio_test[test_columns]
# test_path = f"/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/combined_data/75_tuntun_api_data_with_features_v3-a_test.csv"
# test_columns=["date", "tic", "close", "high", "low", "volume"]
# df_test.to_csv(test_path, index=False)
# print(f"DF TEST IS SAVED TO: {test_path}")
# exit()

from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv

MODE = "train"
checkpoint_dir = ""
if MODE == "train":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"eiie_{timestamp}"
    main_dir = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/train"
    d = f"{main_dir}/v2_result/{model_name}"
    checkpoint_dir = f"{d}/checkpoint"
    os.makedirs(d, exist_ok=True)

features=[
    "close",
    "high",
    "low",
    "volume",
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    # "cci_30", # RESULTED IN NAN
    # "dx_30", # RESULTED IN NAN
    "close_30_sma"
    # "close_60_sma" # RESULTED IN NAN
]
initial_features = len(features)
initial_amount = 1e9
comission_fee_model = "trf_v2"
# comission_fee_pct = 0.0025
buying_fee_pct = 0.0025
selling_fee_pct = 0.0025
time_column = "date"
tics_in_portfolio = "all"
normalize_df = "by_previous_time"
reward_scaling = 1
alpha = 1
use_sortino_ratio = True
risk_free_rate = 0.05
save_detailed_step = 2

# detailed_actions_file = f"{d}/result_eiie_{MODE}.xlsx"
detailed_actions_file = checkpoint_dir
environment_train = PortfolioOptimizationEnv(
        df_portfolio_train,
        initial_amount=initial_amount,
        comission_fee_model=comission_fee_model,
        # comission_fee_pct=comission_fee_pct,
        buying_fee_pct=buying_fee_pct,
        selling_fee_pct=selling_fee_pct,
        time_window=TIME_WINDOW,
        features=features,
        time_column=time_column,
        normalize_df=normalize_df,
        tics_in_portfolio=tics_in_portfolio,
        alpha=alpha,
        mode="train",
        use_sortino_ratio=use_sortino_ratio,
        risk_free_rate=risk_free_rate,
        detailed_actions_file=detailed_actions_file,
        save_detailed_step=save_detailed_step,
        reward_scaling=reward_scaling,
    )
environment_dev = PortfolioOptimizationEnv(
        df_portfolio_dev,
        initial_amount=initial_amount,
        comission_fee_model=comission_fee_model,
        # comission_fee_pct=comission_fee_pct,
        buying_fee_pct=buying_fee_pct,
        selling_fee_pct=selling_fee_pct,
        time_window=TIME_WINDOW,
        features=features,
        time_column=time_column,
        normalize_df=normalize_df,
        tics_in_portfolio="all",
        alpha=alpha,
        mode="dev",
        use_sortino_ratio=use_sortino_ratio,
        risk_free_rate=risk_free_rate,
        detailed_actions_file=detailed_actions_file,
        save_detailed_step=save_detailed_step,
        reward_scaling=reward_scaling,
    )
environment_test = PortfolioOptimizationEnv(
        df_portfolio_test,
        initial_amount=initial_amount,
        comission_fee_model=comission_fee_model,
        # comission_fee_pct=comission_fee_pct,
        buying_fee_pct=buying_fee_pct,
        selling_fee_pct=selling_fee_pct,
        time_window=TIME_WINDOW,
        features=features,
        time_column="date",
        normalize_df=normalize_df,
        tics_in_portfolio="all",
        alpha=alpha,
        mode="test",
        use_sortino_ratio=use_sortino_ratio,
        risk_free_rate=risk_free_rate,
        detailed_actions_file=detailed_actions_file,
        save_detailed_step=save_detailed_step,
        reward_scaling=reward_scaling,
    )
train_env_conf = {
    "train_range": f"{TRAIN_START_DATE}_{TRAIN_END_DATE}",
    "dev_range": f"{DEV_START_DATE}_{DEV_END_DATE}",
    "test_range": f"{TEST_START_DATE}_{TEST_END_DATE}",
    "initial_amount": initial_amount,
    "comission_fee_model": comission_fee_model,
    # "comission_fee_pct": comission_fee_pct,
    "buying_fee_pct": buying_fee_pct,
    "selling_fee_pct": selling_fee_pct,
    "time_window": TIME_WINDOW,
    "features": features,
    "time_column": time_column,
    "normalize_df": normalize_df,
    "tics_in_portfolio": tics_in_portfolio,
    "reward_scaling": reward_scaling,
    "alpha": alpha,
    "mode": MODE,
    "use_sortino_ratio": use_sortino_ratio,
    "risk_free_rate": risk_free_rate,
    "save_detailed_step": save_detailed_step,
    "detailed_actions_file": detailed_actions_file
}
if MODE == "train":
    train_env_str = json.dumps(train_env_conf, indent=3)
    print(train_env_str)
    with open(f"{d}/{MODE}_env_conf.json", "w+") as f:
        f.write(train_env_str)

from finrl.agents.portfolio_optimization.architectures import EIIE
from finrl.agents.portfolio_optimization.models import DRLAgent
import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)


lr = 0.01
action_noise = 0.1
episodes = 10
policy_str = "EIIE"
use_reward_in_loss = True
model_kwargs = {
    "lr": lr,
    "policy": EIIE,
    "action_noise": action_noise,
    "checkpoint_dir": checkpoint_dir,
    "policy_str": policy_str,
    "use_reward_in_loss": use_reward_in_loss
}
model_kwargs_str = {
    "lr": lr,
    "policy": policy_str,
    "action_noise": action_noise,
    "episodes": episodes,
    "use_reward_in_loss": use_reward_in_loss
}
policy_kwargs = {
    "initial_features": initial_features,
    "k_size": 3,
    "conv_mid_features": 2,
    "conv_final_features": 20,
    "time_window": TIME_WINDOW
}
model_conf = {
    "model_kwargs": model_kwargs_str,
    "policy_kwargs": policy_kwargs
}
if MODE == "train":
    model_conf_str = json.dumps(model_conf, indent=3)
    print(model_conf_str)
    with open(f"{d}/model_conf.json", "w+") as f:
        f.write(model_conf_str)

model = DRLAgent(environment_train, dev_env=environment_dev, test_env=environment_test).get_model("pg", device, model_kwargs, policy_kwargs)
if MODE == "train":
    DRLAgent.train_model(model, episodes=episodes)

# d = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/train/v2_result/backup/eiie_20241214_080627_alpha_0.1"
# d = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/train/v2_result/backup/eiie_20241214_082250_alpha_0.2"
# d = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/train/v2_result/eiie_20241220_095522_profitable_in_train_nan_in_dev_and_test"
# d = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/train/v2_result/eiie_20241223_174514_graph_ep10/checkpoint-10"
# d = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/train/v2_result/eiie_20241223_174553_graph_ep100/checkpoint-100"
# d = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/train/v2_result/eiie_20241224_145508_test_profitable_ep10_time_window_5/checkpoint-10"
# model_path = f"{d}/policy_EIIE_10.pt"

# if MODE == "train":
#     torch.save(model.train_policy.state_dict(), model_path)

# instantiate an architecture with the same arguments used in training
# and load with load_state_dict.
policy = None
if MODE == "train":
    policy = model.train_policy
else:
    print(f"Currently not training, loading the model from:\n{model_path}\n")
    policy = EIIE(**policy_kwargs)
    policy.load_state_dict(torch.load(model_path))

# dev
# MODE = "dev"
# detailed_actions_file = f"{d}/result_eiie_{MODE}.xlsx"
# # detailed_actions_file = checkpoint_dir
# environment_dev = PortfolioOptimizationEnv(
#         df_portfolio_dev,
#         initial_amount=initial_amount,
#         comission_fee_model=comission_fee_model,
#         # comission_fee_pct=comission_fee_pct,
#         buying_fee_pct=buying_fee_pct,
#         selling_fee_pct=selling_fee_pct,
#         time_window=TIME_WINDOW,
#         features=features,
#         time_column=time_column,
#         normalize_df=normalize_df,
#         tics_in_portfolio="all",
#         alpha=alpha,
#         mode="dev",
#         use_sortino_ratio=use_sortino_ratio,
#         risk_free_rate=risk_free_rate,
#         detailed_actions_file=detailed_actions_file,
#     )
# DRLAgent.DRL_validation(model, environment_dev, policy=policy)

# testing
# MODE = "test"
# detailed_actions_file = f"{d}/result_eiie_{MODE}.xlsx"
# # detailed_actions_file = checkpoint_dir
# environment_test = PortfolioOptimizationEnv(
#         df_portfolio_test,
#         initial_amount=initial_amount,
#         comission_fee_model=comission_fee_model,
#         # comission_fee_pct=comission_fee_pct,
#         buying_fee_pct=buying_fee_pct,
#         selling_fee_pct=selling_fee_pct,
#         time_window=TIME_WINDOW,
#         features=features,
#         time_column="date",
#         normalize_df=normalize_df,
#         tics_in_portfolio="all",
#         alpha=alpha,
#         mode=MODE,
#         use_sortino_ratio=use_sortino_ratio,
#         risk_free_rate=risk_free_rate,
#         detailed_actions_file=detailed_actions_file,
#     )
# DRLAgent.DRL_validation(model, environment_test, policy=policy)
