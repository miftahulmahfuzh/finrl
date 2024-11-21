import logging
logging.getLogger('matplotlib.font_manager').disabled = True
import torch

import numpy as np
import pandas as pd

from torch_geometric.utils import k_hop_subgraph

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv
from finrl.agents.portfolio_optimization.models import DRLAgent
from finrl.agents.portfolio_optimization.architectures import GPM

device = "cpu"
if torch.cuda.is_available():
    print("CUDA AVAILABLE")
    device = 'cuda:0'
else:
    print("CUDA IS NOT AVAILABLE")
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
d = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/Temporal_Relational_Stock_Ranking_FinRL"
fname = f"{d}/temporal_data/NASDAQ_temporal_data.csv"
nasdaq_temporal = pd.read_csv(fname)
print(f"NASDAQ TEMPORAL")
print(nasdaq_temporal.head())

fname = f"{d}/relational_data/edge_indexes/NASDAQ_sector_industry_edge_index.npy"
nasdaq_edge_index = np.load(fname)
print("\nNASDAQ EDGE INDEX")
print(nasdaq_edge_index)

nasdaq_edge_type = np.load(f"{d}/relational_data/edge_types/NASDAQ_sector_industry_edge_type.npy")
print("\nNASDAQ EDGE TYPE")
print(nasdaq_edge_type)

list_of_stocks = nasdaq_temporal["tic"].unique().tolist()
tics_in_portfolio = ["AAPL", "CMCSA", "CSCO", "FB", "HBAN", "INTC", "MSFT", "MU", "NVDA", "QQQ", "XIV"]

portfolio_nodes = []
for tic in tics_in_portfolio:
    portfolio_nodes.append(list_of_stocks.index(tic))
print("PORTFOLIO NODES")
print(portfolio_nodes)

nodes_kept, new_edge_index, nodes_to_select, edge_mask = k_hop_subgraph(
    torch.LongTensor(portfolio_nodes),
    2,
    torch.from_numpy(nasdaq_edge_index),
    relabel_nodes=True,
)

# reduce temporal data
nodes_kept = nodes_kept.tolist()
nasdaq_temporal["tic_id"], _ = pd.factorize(nasdaq_temporal["tic"], sort=True)
nasdaq_temporal = nasdaq_temporal[nasdaq_temporal["tic_id"].isin(nodes_kept)]
nasdaq_temporal = nasdaq_temporal.drop(columns="tic_id")
print("\nREDUCED NASDAQ TEMPORAL")
print(nasdaq_temporal)

# reduce edge type
new_edge_type = torch.from_numpy(nasdaq_edge_type)[edge_mask]
_, new_edge_type = torch.unique(new_edge_type, return_inverse=True)
print("\nREDUCED EDGE TYPE")
print(new_edge_type)

df_portfolio = nasdaq_temporal[["day", "tic", "close", "high", "low"]]

df_portfolio_train = df_portfolio[df_portfolio["day"] < 979]
df_portfolio_test = df_portfolio[df_portfolio["day"] >= 979]

environment_train = PortfolioOptimizationEnv(
        df_portfolio_train,
        initial_amount=100000,
        comission_fee_pct=0.0025,
        time_window=50,
        features=["close", "high", "low"],
        time_column="day",
        normalize_df=None, # dataframe is already normalized
        tics_in_portfolio=tics_in_portfolio
    )

environment_test = PortfolioOptimizationEnv(
        df_portfolio_test,
        initial_amount=100000,
        comission_fee_pct=0.0025,
        time_window=50,
        features=["close", "high", "low"],
        time_column="day",
        normalize_df=None, # dataframe is already normalized
        tics_in_portfolio=tics_in_portfolio
    )

model_kwargs = {
    "lr": 0.01,
    "policy": GPM,
}

# here, we can set GPM's parameters
policy_kwargs = {
    "edge_index": new_edge_index,
    "edge_type": new_edge_type,
    "nodes_to_select": nodes_to_select
}

model = DRLAgent(environment_train).get_model("pg", device, model_kwargs, policy_kwargs)
DRLAgent.train_model(model, episodes=2)
