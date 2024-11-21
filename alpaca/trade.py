from alpaca_paper_trading import AlpacaPaperTrading
from finrl.config_tickers import DOW_30_TICKER
from finrl.config import (
    INDICATORS,
    ERL_PARAMS,
)

ERL_PARAMS = {"learning_rate": 3e-6,"batch_size": 2048,"gamma":  0.985,
        "seed":312,"net_dimension":[128,64], "target_step":5000, "eval_gap":30,
        "eval_times":1}

ticker_list = DOW_30_TICKER
print(f"TICKER LIST: {ticker_list}")
action_dim = len(DOW_30_TICKER)
print(f"INDICATORS: {INDICATORS}")
state_dim = 1 + 2 + 3 * action_dim + len(INDICATORS) * action_dim
print(f"STATE DIM: {state_dim}")

API_KEY = "PKSCD25SIIWWGDU6MA2S"
API_SECRET = "cm90rIwfwi6kggGJiV7aeviac05wuKeKCLijifEt"
API_BASE_URL = 'https://paper-api.alpaca.markets'

paper_trading_erl = AlpacaPaperTrading(
        ticker_list = DOW_30_TICKER,
        time_interval = '1Min',
        drl_lib = 'elegantrl',
        agent = 'ppo',
        cwd = './papertrading_erl',
        net_dim = ERL_PARAMS['net_dimension'],
        state_dim = state_dim,
        action_dim= action_dim,
        API_KEY = API_KEY,
        API_SECRET = API_SECRET,
        API_BASE_URL = API_BASE_URL,
        tech_indicator_list = INDICATORS,
        turbulence_thresh=30,
        max_stock=1e2)
paper_trading_erl.run()
