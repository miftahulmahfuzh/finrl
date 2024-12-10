from __future__ import annotations

import gymnasium as gym
import numpy as np
from numpy import random as rd


class StockTradingEnv(gym.Env):
    def __init__(
        self,
        config,
        initial_account=1e6,
        gamma=0.99,
        turbulence_thresh=99,
        min_stock_rate=0.1,
        max_stock=1e2,
        initial_capital=1e6,
        buy_cost_pct=1e-3,
        sell_cost_pct=1e-3,
        reward_scaling=2**-11,
        initial_stocks=None,
    ):
        price_ary = config["price_array"]
        tech_ary = config["tech_array"]
        turbulence_ary = config["turbulence_array"]
        if_train = config["if_train"]
        self.price_ary = price_ary.astype(np.float32)
        self.tech_ary = tech_ary.astype(np.float32)
        self.turbulence_ary = turbulence_ary

        self.tech_ary = self.tech_ary * 2**-7
        self.turbulence_bool = (turbulence_ary > turbulence_thresh).astype(np.float32)
        self.turbulence_ary = (
            self.sigmoid_sign(turbulence_ary, turbulence_thresh) * 2**-5
        ).astype(np.float32)

        stock_dim = self.price_ary.shape[1]
        self.gamma = gamma
        self.max_stock = max_stock
        self.min_stock_rate = min_stock_rate
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.initial_capital = initial_capital
        self.initial_stocks = (
            np.zeros(stock_dim, dtype=np.float32)
            if initial_stocks is None
            else initial_stocks
        )

        # reset()
        self.day = None
        self.amount = None
        self.stocks = None
        self.total_asset = None
        self.gamma_reward = None
        self.initial_total_asset = None

        # environment information
        self.env_name = "StockEnv"
        # self.state_dim = 1 + 2 + 2 * stock_dim + self.tech_ary.shape[1]
        # # amount + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        self.state_dim = 1 + 2 + 3 * stock_dim + self.tech_ary.shape[1]
        # amount + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        self.stocks_cd = None
        self.action_dim = stock_dim
        self.max_step = self.price_ary.shape[0] - 1
        self.if_train = if_train
        self.if_discrete = False
        self.target_return = 10.0
        self.episode_return = 0.0

        self.observation_space = gym.spaces.Box(
            low=-3000, high=3000, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )
        # MIFTAH'S CODE
        self.history_action = {} # detailed tuple (price, action, num_shares) on each tic on a day
        self.history_amount = {} # amount of money the model has on a day
        self.history_total = {} # sum(price_i*num_shares_i) for each tickers, add with amount of money

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        self.day = 0
        price = self.price_ary[self.day]

        if self.if_train:
            self.stocks = (
                self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)
            ).astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = (
                self.initial_capital * rd.uniform(0.95, 1.05)
                - (self.stocks * price).sum()
            )
        else:
            self.stocks = self.initial_stocks.astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = self.initial_capital

        self.total_asset = self.amount + (self.stocks * price).sum()
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0
        return self.get_state(price), {}  # state

    def step(self, actions):
        INFERENCE = False
        if INFERENCE:
            print("\n======================")
            print(f"DAY {self.day}")
        # print(f"PRE ACTIONS: {actions}")
        actions = (actions * self.max_stock).astype(int)
        if INFERENCE:
            print(f"POST ACTIONS: {actions}")

        self.day += 1
        price = self.price_ary[self.day]
        self.stocks_cool_down += 1

        if self.turbulence_bool[self.day] == 0:
            min_action = int(self.max_stock * self.min_stock_rate)  # stock_cd
            # print(f"MIN ACTION: {min_action}")
            sell_indexes = np.where(actions < -min_action)[0]
            if INFERENCE:
                print(self.stocks)
                print(f"SELL INDEXES: {sell_indexes}")
            for index in sell_indexes:  # sell_index:
                if price[index] > 0:  # Sell only if current asset is > 0
                    sell_num_shares = min(self.stocks[index], -actions[index])
                    # if index == 0 and sell_num_shares > 0:
                    #     print(f"INDEX: {index} - SELL NUM SHARES: {sell_num_shares}")
                    self.stocks[index] -= sell_num_shares
                    self.amount += (
                        price[index] * sell_num_shares * (1 - self.sell_cost_pct)
                    )
                    self.stocks_cool_down[index] = 0
            buy_indexes = np.where(actions > min_action)[0]
            if INFERENCE:
                print(f"BUY INDEXES: {buy_indexes}")
            for index in buy_indexes:  # buy_index:
                if (
                    price[index] > 0
                ):  # Buy only if the price is > 0 (no missing data in this particular date)
                    buy_num_shares = min(self.amount // price[index], actions[index])
                    self.stocks[index] += buy_num_shares
                    self.amount -= (
                        price[index] * buy_num_shares * (1 + self.buy_cost_pct)
                    )
                    self.stocks_cool_down[index] = 0
            trade_indexes = np.concatenate((sell_indexes, buy_indexes))
            hold_indexes = [i for i in range(len(price)) if i not in trade_indexes]
            if INFERENCE:
                print(f"HOLD INDEXES: {hold_indexes}")

        else:  # sell all when turbulence
            if INFERENCE:
                print(f"SELL ALL STOCKS BECAUSE OF TURBULENCE")
                print(self.stocks)
            self.amount += (self.stocks * price).sum() * (1 - self.sell_cost_pct)
            self.stocks[:] = 0
            self.stocks_cool_down[:] = 0

        # added by miftah on 02-12-2024
        self.history_action[self.day] = [(int(tic_price), int(tic_action), int(tic_stock)) for tic_price, tic_action, tic_stock in zip(price, actions, self.stocks)]
        self.history_amount[self.day] = int(self.amount)
        tmp_total = sum([(int(tic_price) * int(tic_stock)) for tic_price, tic_stock in zip(price, self.stocks)])
        self.history_total[self.day] = tmp_total + int(self.amount)
        if INFERENCE:
            print(f"AMOUNT OF MONEY AFTER MARKET CLOSE: {self.amount}")

        state = self.get_state(price)
        total_asset = self.amount + (self.stocks * price).sum()
        reward = (total_asset - self.total_asset) * self.reward_scaling
        self.total_asset = total_asset

        self.gamma_reward = self.gamma_reward * self.gamma + reward
        done = self.day == self.max_step
        if done:
            reward = self.gamma_reward
            self.episode_return = total_asset / self.initial_total_asset

        return state, reward, done, False, dict()

    def get_state(self, price):
        amount = np.array(self.amount * (2**-12), dtype=np.float32)
        scale = np.array(2**-6, dtype=np.float32)
        return np.hstack(
            (
                amount,
                self.turbulence_ary[self.day],
                self.turbulence_bool[self.day],
                price * scale,
                self.stocks * scale,
                self.stocks_cool_down,
                self.tech_ary[self.day],
            )
        )  # state.astype(np.float32)

    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh
