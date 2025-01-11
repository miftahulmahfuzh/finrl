"""From FinRL https://github.com/AI4Finance-LLC/FinRL/tree/master/finrl/env"""

from __future__ import annotations

import math

import gym
import matplotlib
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from pathlib import Path

# added by miftah on 11-12-2024
import os
from datetime import datetime as dt, timedelta
from scipy.optimize import newton  # Importing the root-finding method

try:
    import quantstats as qs
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        """QuantStats module not found, environment can't plot results and calculate indicadors.
        This module is not installed with FinRL. Install by running one of the options:
        pip install quantstats --upgrade --no-cache-dir
        conda install -c ranaroussi quantstats
        """
    )


class PortfolioOptimizationEnv(gym.Env):
    """A portfolio allocation environment for OpenAI gym.

    This environment simulates the interactions between an agent and the financial market
    based on data provided by a dataframe. The dataframe contains the time series of
    features defined by the user (such as closing, high and low prices) and must have
    a time and a tic column with a list of datetimes and ticker symbols respectively.
    An example of dataframe is shown below::

            date        high            low             close           tic
        0   2020-12-23  0.157414        0.127420        0.136394        ADA-USD
        1   2020-12-23  34.381519       30.074295       31.097898       BNB-USD
        2   2020-12-23  24024.490234    22802.646484    23241.345703    BTC-USD
        3   2020-12-23  0.004735        0.003640        0.003768        DOGE-USD
        4   2020-12-23  637.122803      560.364258      583.714600      ETH-USD
        ... ...         ...             ...             ...             ...

    Based on this dataframe, the environment will create an observation space that can
    be a Dict or a Box. The Box observation space is a three-dimensional array of shape
    (f, n, t), where f is the number of features, n is the number of stocks in the
    portfolio and t is the user-defined time window. If the environment is created with
    the parameter return_last_action set to True, the observation space is a Dict with
    the following keys::

        {
        "state": three-dimensional Box (f, n, t) representing the time series,
        "last_action": one-dimensional Box (n+1,) representing the portfolio weights
        }

    Note that the action space of this environment is an one-dimensional Box with size
    n + 1 because the portfolio weights must contains the weights related to all the
    stocks in the portfolio and to the remaining cash.

    Attributes:
        action_space: Action space.
        observation_space: Observation space.
        episode_length: Number of timesteps of an episode.
        portfolio_size: Number of stocks in the portfolio.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        initial_amount,
        order_df=True,
        return_last_action=False,
        normalize_df="by_previous_time",
        reward_scaling=1,
        comission_fee_model="trf",
        features=["close", "high", "low"],
        comission_fee_pct=0,
        valuation_feature="close",
        time_column="date",
        time_format="%Y-%m-%d",
        tic_column="tic",
        tics_in_portfolio="all",
        time_window=1,
        cwd="./",
        new_gym_api=False,
        # below is added by miftah
        use_sell_indicator=None,
        sell_indicator_column="",
        turbulence_threshold=99,
        take_profit_threshold_1=.8,
        take_profit_threshold_2=.05,
        stop_loss_threshold=.05,
        buying_fee_pct=0,
        selling_fee_pct=0,
        alpha=.01,
        mode="train", # options: train, dev, test
        use_sortino_ratio=False,
        risk_free_rate=0,
        cycle_length=1, # do reallocation for every cycle_length days
        detailed_actions_file="",
        eval_episode=0, # what training episode is the evaluation (dev, test) currently on
        save_detailed_step=25 # write detailed action xlsx for each save_detailed_step
    ):
        """Initializes environment's instance.

        Args:
            df: Dataframe with market information over a period of time.
            initial_amount: Initial amount of cash available to be invested.
            order_df: If True input dataframe is ordered by time.
            return_last_action: If True, observations also return the last performed
                action. Note that, in that case, the observation space is a Dict.
            normalize_df: Defines the normalization method applied to input dataframe.
                Possible values are "by_previous_time", "by_fist_time_window_value",
                "by_COLUMN_NAME" (where COLUMN_NAME must be changed to a real column
                name) and a custom function. If None no normalization is done.
            reward_scaling: A scaling factor to multiply the reward function. This
                factor can help training.
            comission_fee_model: Model used to simulate comission fee. Possible values
                are "trf" (for transaction remainder factor model) and "wvm" (for weights
                vector modifier model). If None, commission fees are not considered.
            comission_fee_pct: Percentage to be used in comission fee. It must be a value
                between 0 and 1.
            features: List of features to be considered in the observation space. The
                items of the list must be names of columns of the input dataframe.
            valuation_feature: Feature to be considered in the portfolio value calculation.
            time_column: Name of the dataframe's column that contain the datetimes that
                index the dataframe.
            time_format: Formatting string of time column.
            tic_name: Name of the dataframe's column that contain ticker symbols.
            tics_in_portfolio: List of ticker symbols to be considered as part of the
                portfolio. If "all", all tickers of input data are considered.
            time_window: Size of time window.
            cwd: Local repository in which resulting graphs will be saved.
            new_gym_api: If True, the environment will use the new gym api standard for
                step and reset methods.
        """
        self._time_window = time_window
        self._time_index = time_window - 1
        self._time_column = time_column
        self._time_format = time_format
        self._tic_column = tic_column
        self._df = df
        self._initial_amount = initial_amount
        self._return_last_action = return_last_action
        self._reward_scaling = reward_scaling
        self._comission_fee_pct = comission_fee_pct
        self._comission_fee_model = comission_fee_model
        self._features = features
        self._valuation_feature = valuation_feature
        self._cwd = Path(cwd)
        self._new_gym_api = new_gym_api

        # below parameters is added by miftah
        self._raw_df = df
        self._raw_df[self._time_column] = pd.to_datetime(self._raw_df[self._time_column])
        self._use_sell_indicator = use_sell_indicator # 'turbulence' / 'stoploss_and_takeprofit'
        self._sell_indicator_column = sell_indicator_column
        self._turbulence_threshold = turbulence_threshold
        self._stop_loss_threshold = stop_loss_threshold
        self._take_profit_threshold_1 = take_profit_threshold_1
        self._take_profit_threshold_2 = take_profit_threshold_2
        self._latest_profit = 0
        self._max_profit = -1
        self._turbulence = 0
        self._trading_date = None
        self._detailed_actions_memory = []
        self._start_timestamp = None
        self._mode = mode
        self.detailed_actions_file = detailed_actions_file
        self._transaction_cost = 0
        self._alpha = alpha # how big is the impact of the transaction cost penalty on reward calculation
        # self._prev_weights = None
        self._first_episode = True
        self._use_sortino_ratio = use_sortino_ratio
        self._risk_free_rate = risk_free_rate
        self._cycle_length = cycle_length
        self._buying_fee_pct = buying_fee_pct
        self._selling_fee_pct = selling_fee_pct
        self._buying_cost = 0.0  # Initialize buying cost
        self._selling_cost = 0.0  # Initialize selling cost
        self._eval_episode = eval_episode
        self._interim_portfolio_value = 0
        self._save_detailed_step = save_detailed_step

        # results file
        self._results_file = self._cwd / "results" / "rl"
        self._results_file.mkdir(parents=True, exist_ok=True)

        # initialize price variation
        self._df_price_variation = None

        # preprocess data
        self._preprocess_data(order_df, normalize_df, tics_in_portfolio)

        # dims and spaces
        self._tic_list = sorted(self._df[self._tic_column].unique())
        self.portfolio_size = (
            len(self._tic_list)
            if tics_in_portfolio == "all"
            else len(tics_in_portfolio)
        )
        action_space = 1 + self.portfolio_size

        # added by miftah
        self._current_stock_prices = [0] * len(self._tic_list)
        tmp = [0] * len(self._tic_list)
        self._prev_weights = np.asarray([1] + tmp)

        # sort datetimes and define episode length
        self._sorted_times = sorted(set(self._df[time_column]))
        self.episode_length = len(self._sorted_times) - time_window + 1

        # define action space
        self.action_space = spaces.Box(low=0, high=1, shape=(action_space,))

        # define observation state
        if self._return_last_action:
            # if  last action must be returned, a dict observation
            # is defined
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(
                            len(self._features),
                            len(self._tic_list),
                            self._time_window,
                        ),
                    ),
                    "last_action": spaces.Box(low=0, high=1, shape=(action_space,)),
                }
            )
        else:
            # if information about last action is not relevant,
            # a 3D observation space is defined
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(self._features), len(self._tic_list), self._time_window),
            )

        self._reset_memory()

        self._portfolio_value = self._initial_amount
        self._terminal = False

    def calculate_duration(self, start: dt, end: dt):
        # Calculate the duration
        duration = end - start

        # Extract total seconds from duration
        total_seconds = int(duration.total_seconds())
        hours, remainder = divmod(abs(total_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)

        # Create the dataframe with the required format
        data = {
            "start": [start.strftime("%Y-%m-%d_%H:%M:%S")],
            "end": [end.strftime("%Y-%m-%d_%H:%M:%S")],
            "duration_hour": [hours],
            "duration_minute": [minutes],
            "duration_second": [seconds],
        }
        return pd.DataFrame(data)

    def step(self, actions, episode=0):
        """Performs a simulation step.

        Args:
            actions: An unidimensional array containing the new portfolio
                weights.

        Note:
            If the environment was created with "return_last_action" set to
            True, the next state returned will be a Dict. If it's set to False,
            the next state will be a Box. You can check the observation state
            through the attribute "observation_space".

        Returns:
            If "new_gym_api" is set to True, the following tuple is returned:
            (state, reward, terminal, truncated, info). If it's set to False,
            the following tuple is returned: (state, reward, terminal, info).

            state: Next simulation state.
            reward: Reward related to the last performed action.
            terminal: If True, the environment is in a terminal state.
            truncated: If True, the environment has passed it's simulation
                time limit. Currently, it's always False.
            info: A dictionary containing informations about the last state.
        """
        # self._terminal = self._time_index >= len(self._sorted_times) - 1
        self._terminal = self._time_index >= len(self._sorted_times) - self._cycle_length

        if self._terminal:
            if self._mode in ["dev", "test"]:
                episode = self._eval_episode
            elif not episode:
                episode = 0
            if int(episode) % self._save_detailed_step == 0:
                print(f"TERMINAL. WRITE DETAILED RESULT {self._mode} FOR EPISODE: {episode}")
                d = f"{self.detailed_actions_file}-{episode}"
                os.makedirs(d, exist_ok=True)
                actions_df = pd.DataFrame(self._detailed_actions_memory)
                print(actions_df)
                self._end_timestamp = dt.now()
                duration_df = self.calculate_duration(self._start_timestamp, self._end_timestamp)
                tmp = self.detailed_actions_file
                tmp = f"{d}/result_eiie_{self._mode}.xlsx"
                if self._mode == "train":
                    tmp = f"{d}/result_eiie_{self._mode}_{episode}.xlsx"
                with pd.ExcelWriter(tmp) as writer:
                    actions_df.to_excel(writer, sheet_name='detailed_actions', index=False)
                    duration_df.to_excel(writer, sheet_name='duration', index=False)
                print(f"Detailed actions and duration is saved to:\n{tmp}")
            else:
                print(f"TERMINAL. SKIP WRITING DETAILED RESULT FOR MODE: {self._mode}, EPISODE: {episode}")

            if self._new_gym_api:
                return self._state, self._reward, self._terminal, False, self._info
            return self._state, self._reward, self._terminal, self._info

        else:
            # transform action to numpy array (if it's a list)
            actions = np.array(actions, dtype=np.float32)
            if np.isnan(actions).any():
                print(f"ACTIONS: {actions}")
                raise ValueError("Array 'actions' contains NaN values.")
            date_str = self._info["end_time"].strftime("%Y-%m-%d")

            # if necessary, normalize weights
            if math.isclose(np.sum(actions), 1, abs_tol=1e-6) and np.min(actions) >= 0:
                weights = actions
            else:
                weights = self._softmax_normalization(actions)

            # POSTPROCESSING WEIGHTS - MIFTAH
            # 1. Round weights to multiplies of 0.05
            weights = np.round(weights * 20) / 20
            # if after this command the sum is zero 0, then we reset the value back
            # 2. Set weights < 0.05 to 0
            weights[weights < 0.05] = 0
            # Renormalize weights to sum to 1 after zeroing small weights
            if np.sum(weights) == 0:
                weights[0] = 1.0
            else:
                weights = weights / np.sum(weights)

            # load next state
            # self._time_index += 1
            self._time_index += self._cycle_length
            self._state, self._info, self._turbulence, self._current_stock_prices = self._get_state_and_info_from_time_index(
                self._time_index
            )

            # USING TURBULENCE AS SELL INDICATOR - MIFTAH
            sell_all = False
            # print(f"TURBULENCE: {self._turbulence}")
            if self._use_sell_indicator == "turbulence":
                if self._turbulence > self._turbulence_threshold:
                    sell_all = True
                if sell_all:
                    # print(f"SELL ALL: {sell_all}\nTURBULENCE: {self._turbulence}\nTHRESHOLD: {self._sell_indicator_threshold}")
                    tmp = [0] * len(self._info["tics"].tolist())
                    weights = [1] + tmp
                    weights = np.asarray(weights)

            # CALCULATE TODAY'S PORTFOLIO USING WEIGHTS ON PREV DAY - MIFTAH
            self._asset_memory["initial"].append(self._portfolio_value)
            if self._first_episode:
                tmp = [0] * len(self._info["tics"])
                self._prev_weights = [1] + tmp
                self._first_episode = False
            portfolio = self._portfolio_value * (self._prev_weights * self._price_variation)
            # so, i moved self._portfolio_value here so it will not reset the value reduction of cost_fee_model calculation
            # previously, this line is in line 487
            self._portfolio_value = np.sum(portfolio)

            self._interim_portfolio_value = np.sum(portfolio)
            reallocate_today = True
            if self._use_sell_indicator == "stoploss_and_takeprofit":
                if len(self._asset_memory["final"]) > 1:
                    first_day_portfolio = self._asset_memory["final"][1]
                    # STOP_LOSS_RULE
                    min_value_to_stop_loss = (1-self._stop_loss_threshold) * first_day_portfolio
                    stop_loss = self._interim_portfolio_value < min_value_to_stop_loss

                    # if total_price is 5% lower than the portfolio value of the previous day,
                    # then we hold funds (dont do transactions on this day)
                    if stop_loss:
                        # print(f"MIN_VALUE: {min_value}\nINTERIM: {self._interim_portfolio_value}")
                        # print(f"PREV_WEIGHTS: {self._prev_weights}")
                        # print(f"WEIGHTS: {weights}\n")
                        weights = self._prev_weights
                        # reallocate_today = False
                    # else:
                        # print(f"IT IS SAVE TO TRADE")
                        # print(f"MIN_VALUE: {min_value}\nINTERIM: {self._interim_portfolio_value}")
                        # print(f"PREV_WEIGHTS: {self._prev_weights}")
                        # print(f"WEIGHTS: {weights}\n")

                    # TAKE_PROFIT_RULE - v1
                    # self._latest_profit = self._interim_portfolio_value - first_day_portfolio
                    # min_value_to_take_profit = self._take_profit_threshold * self._max_profit
                    # take_profit = self._latest_profit >= min_value_to_take_profit
                    # # if the latest_profit minimum reached 80% of the max profit,
                    # # then we do trading on that day (we use the model actions), otherwise, we hold
                    # # print(f"TRADING DATE: {self._trading_date}")
                    # # print(f"MAX_PROFIT: {self._max_profit}")
                    # # print(f"LATEST PROFIT: {latest_profit}")
                    # if take_profit:
                    #     # print(f"LATEST PROFIT REACHED 80% OF MAX_PROFIT\n")
                    #     # print(f"WE REALLOCATE STOCKS TODAY")
                    #     pass
                    # else:
                    #     # print(f"LATEST PROFIT IS NOT ENOUGH. AVOIDING TRANSACTIONS FOR TODAY")
                    #     weights = self._prev_weights
                    #     reallocate_today = False
                    # self._max_profit = max(self._latest_profit, self._max_profit)

                    # TAKE_PROFIT_RULE - v2
                    self._latest_profit = self._interim_portfolio_value - first_day_portfolio
                    self._max_profit = max(self._latest_profit, self._max_profit)
                    a = self._max_profit > (1 + self._take_profit_threshold_2) * first_day_portfolio
                    b = self._latest_profit < self._take_profit_threshold_1 * self._max_profit
                    take_profit = a and b
                    if take_profit:
                        # if take_profit==True, we sell everything today
                        tmp = [0] * len(self._info["tics"])
                        weights = [1] + tmp
                        weights = np.asarray(weights)
                        # reallocate_today = True

                    reallocate_today = stop_loss or take_profit

            # save initial portfolio weights for this time step
            self._actions_memory.append(weights)

            if self._comission_fee_model == "trf_v2":
                # **TRF_V2 SECTION START - MIFTAH**
                self._selling_cost = 0
                self._buying_cost = 0

                buying_fee_pct = self._buying_fee_pct  # Separate buying fee percentage
                selling_fee_pct = self._selling_fee_pct  # Separate selling fee percentage
                # Initialize mu considering both buying and selling fees
                mu = 1.0 - buying_fee_pct - selling_fee_pct

                if reallocate_today:
                    # Calculate w_i based on price variation
                    ltmp = self._prev_weights * self._price_variation
                    w_i = np.zeros_like(ltmp)
                    if np.sum(ltmp) > 0:
                        w_i = ltmp / np.sum(ltmp)

                    selling_reduction = np.maximum(w_i[1:] - weights[1:], 0)
                    selling_cost = selling_fee_pct * np.sum(selling_reduction)
                    buying_increase = np.maximum(weights[1:] - w_i[1:], 0)
                    buying_cost = buying_fee_pct * np.sum(buying_increase)

                    # Assign separate transaction costs
                    self._selling_cost = selling_cost
                    self._buying_cost = buying_cost

                # Calculate total transaction cost
                self._transaction_cost = self._selling_cost + self._buying_cost

                # Update portfolio value
                self._info["trf_mu"] = mu
                self._portfolio_value = self._portfolio_value * (1 - self._transaction_cost)

                # **TRF_V2 SECTION END**

            # save initial portfolio value of this time step
            # self._asset_memory["initial"].append(self._portfolio_value)
            self._prev_weights = weights

            date_str = self._info["end_time"].strftime("%Y-%m-%d")
            daily_log = {
                "date": date_str,
                "portfolio": self._portfolio_value,
                "interim_portfolio": self._interim_portfolio_value,
                # "turbulence": self._turbulence,
                "today_profit": self._latest_profit,
                "max_profit": self._max_profit,
                "buying_cost": self._buying_cost,
                "selling_cost": self._selling_cost,
                "transaction_cost": self._transaction_cost,
                "reallocate_today": reallocate_today,
            }
            # if self._use_sell_indicator != "stoploss_and_takeprofit":
            #     daily_log.pop("today_profit")
            #     daily_log.pop("max_profit")

            # CALCULATE WEIGHTS PERCENT - MIFTAH
            weights_percent = weights * 100
            weights_percent = np.asarray(weights_percent)
            weights_percent[weights_percent < 0.01] = 0  # Set small weights to zero
            weight_details = ["CASH"] + self._info["tics"] # add column for cash allocation
            for i, ticker in enumerate(weight_details):
                daily_log[ticker] = weights_percent[i]
            for i, ticker in enumerate(weight_details):
                daily_log[f"{ticker}_v"] = self._price_variation[i]

            # calculate new portfolio value and weights
            # self._portfolio_value = np.sum(portfolio) # MOVED TO LINE 371
            weights = portfolio / self._portfolio_value

            daily_log["portfolio"] = self._portfolio_value
            self._detailed_actions_memory.append(daily_log)

            # save final portfolio value and weights of this time step
            self._asset_memory["final"].append(self._portfolio_value)
            self._final_weights.append(weights)

            # save date memory
            self._date_memory.append(self._info["end_time"])

            # Calculate Sortino Ratio as the reward - MIFTAH
            # First, calculate returns
            returns = np.array(self._portfolio_return_memory)

            # If we don't have enough returns to calculate, use a default reward
            portfolio_reward = 0
            if self._use_sortino_ratio:
                if len(returns) >= 2:
                    # Calculate mean return
                    mean_return = np.mean(returns)

                    # Calculate downside deviation (standard deviation of negative returns)
                    downside_returns = returns[returns < 0]

                    # Handle case with no downside returns
                    if len(downside_returns) == 0:
                        downside_dev = 0
                    else:
                        downside_dev = np.std(downside_returns)

                    # Sortino Ratio calculation
                    # Use risk-free rate as 0 for simplicity
                    # Sortino Ratio = (Mean Portfolio Return - Risk-Free Rate) / Downside Deviation
                    sortino_ratio = (mean_return - self._risk_free_rate) / (downside_dev if downside_dev > 0 else 1e-10)

                    # Use Sortino Ratio as the reward
                    portfolio_reward = sortino_ratio
            #######################################################

            # define portfolio return
            rate_of_return = (
                self._asset_memory["final"][-1] / self._asset_memory["final"][-2]
            )
            portfolio_return = rate_of_return - 1
            # TRANSACTION COST AFFECTS PORTFOLIO REWARD CALCULATION - MIFTAH
            if not self._use_sortino_ratio:
                portfolio_reward = rate_of_return
            portfolio_reward_tmp = portfolio_reward - (self._alpha * self._transaction_cost)
            if portfolio_reward_tmp > 0:
                # portfolio_reward = np.log(portfolio_reward - (self._alpha * self._transaction_cost))
                portfolio_reward = np.log(portfolio_reward_tmp)
            else:
                portfolio_reward = max(portfolio_reward, 0)

            # save portfolio return memory
            self._portfolio_return_memory.append(portfolio_return)
            self._portfolio_reward_memory.append(portfolio_reward)

            # Define portfolio return
            self._reward = portfolio_reward
            self._reward = self._reward * self._reward_scaling

        if self._new_gym_api:
            return self._state, self._reward, self._terminal, False, self._info
        return self._state, self._reward, self._terminal, self._info

    def reset(self):
        """Resets the environment and returns it to its initial state (the
        fist date of the dataframe).

        Note:
            If the environment was created with "return_last_action" set to
            True, the initial state will be a Dict. If it's set to False,
            the initial state will be a Box. You can check the observation
            state through the attribute "observation_space".

        Returns:
            If "new_gym_api" is set to True, the following tuple is returned:
            (state, info). If it's set to False, only the initial state is
            returned.

            state: Initial state.
            info: Initial state info.
        """
        # time_index must start a little bit in the future to implement lookback
        self._time_index = self._time_window - 1
        self._reset_memory()

        self._state, self._info, self._turbulence, self._current_stock_prices = self._get_state_and_info_from_time_index(
            self._time_index
        )
        self._portfolio_value = self._initial_amount
        self._terminal = False

        if self._new_gym_api:
            return self._state, self._info
        return self._state

    def _get_state_and_info_from_time_index(self, time_index):
        """Gets state and information given a time index. It also updates "data"
        attribute with information about the current simulation step.

        Args:
            time_index: An integer that represents the index of a specific datetime.
                The initial datetime of the dataframe is given by 0.

        Note:
            If the environment was created with "return_last_action" set to
            True, the returned state will be a Dict. If it's set to False,
            the returned state will be a Box. You can check the observation
            state through the attribute "observation_space".

        Returns:
            A tuple with the following form: (state, info).

            state: The state of the current time index. It can be a Box or a Dict.
            info: A dictionary with some informations about the current simulation
                step. The dict has the following keys::

            {
                "tics": List of ticker symbols,
                "start_time": Start time of current time window,
                "start_time_index": Index of start time of current time window,
                "end_time": End time of current time window,
                "data": Data related to the current time window,
                "end_time_index": Index of end time of current time window,
                "price_variation": Price variation of current time step
            }
        """
        # returns state in form (channels, tics, timesteps)
        end_time = self._sorted_times[time_index]
        start_time = self._sorted_times[time_index - (self._time_window - 1)]
        # print(f"TIME_INDEX: {time_index}")
        # print(f"START_TIME: {start_time}")
        # print(f"END_TIME: {end_time}\n")
        self._trading_date = end_time

        # define data to be used in this time step
        self._data = self._df[
            (self._df[self._time_column] >= start_time)
            & (self._df[self._time_column] <= end_time)
        ][[self._time_column, self._tic_column] + self._features]

        # define price variation of this time_step
        self._price_variation = self._df_price_variation[
            self._df_price_variation[self._time_column] == end_time
        ][self._valuation_feature].to_numpy()
        self._price_variation = np.insert(self._price_variation, 0, 1)

        # get turbulence value
        turbulence = 0
        if self._use_sell_indicator == "turbulence":
            turbulence = self._df[
                self._df[self._time_column] == end_time
            ][self._sell_indicator_column].iloc[0]

        # get stock price value
        current_stock_prices = {}
        for tic in self._tic_list:
            tic_data = self._raw_df[self._raw_df[self._tic_column] == tic]
            # print(f"TIC_DATA: {tic_data}")
            tic_stock_price = tic_data[
                tic_data[self._time_column] == end_time
            ][self._valuation_feature].iloc[0]
            current_stock_prices[tic] = tic_stock_price
        # print(f"CURRENT STOCK PRICES:\n{current_stock_prices}")

        # define state to be returned
        state = None
        for tic in self._tic_list:
            tic_data = self._data[self._data[self._tic_column] == tic]
            tic_data = tic_data[self._features].to_numpy().T
            tic_data = tic_data[..., np.newaxis]
            state = tic_data if state is None else np.append(state, tic_data, axis=2)
        state = state.transpose((0, 2, 1))
        info = {
            "tics": self._tic_list,
            "start_time": start_time,
            "start_time_index": time_index - (self._time_window - 1),
            "end_time": end_time,
            "end_time_index": time_index,
            "data": self._data,
            "price_variation": self._price_variation,
        }
        return self._standardize_state(state), info, turbulence, current_stock_prices

    def render(self, mode="human"):
        """Renders the environment.

        Returns:
            Observation of current simulation step.
        """
        return self._state

    def _softmax_normalization(self, actions):
        """Normalizes the action vector using softmax function.

        Returns:
            Normalized action vector (portfolio vector).
        """
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output

    def enumerate_portfolio(self):
        """Enumerates the current porfolio by showing the ticker symbols
        of all the investments considered in the portfolio.
        """
        print("Index: 0. Tic: Cash")
        for index, tic in enumerate(self._tic_list):
            print(f"Index: {index + 1}. Tic: {tic}")

    def _preprocess_data(self, order, normalize, tics_in_portfolio):
        """Orders and normalizes the environment's dataframe.

        Args:
            order: If true, the dataframe will be ordered by ticker list
                and datetime.
            normalize: Defines the normalization method applied to the dataframe.
                Possible values are "by_previous_time", "by_fist_time_window_value",
                "by_COLUMN_NAME" (where COLUMN_NAME must be changed to a real column
                name) and a custom function. If None no normalization is done.
            tics_in_portfolio: List of ticker symbols to be considered as part of the
                portfolio. If "all", all tickers of input data are considered.
        """
        # order time dataframe by tic and time
        if order:
            self._df = self._df.sort_values(by=[self._tic_column, self._time_column])
        # defining price variation after ordering dataframe
        pv_path = f"/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/combined_data/75_tic_price_variation_v5_{self._mode}_9_features.csv"
        self._df_price_variation = None
        if os.path.isfile(pv_path):
            self._df_price_variation = pd.read_csv(pv_path)
            print(f"\nLOADED PRICE VARIATION DATA FROM:\n{pv_path}")
        else:
            self._df_price_variation = self._temporal_variation_df()
            self._df_price_variation.to_csv(pv_path, index=False)
            print(f"\nSAVED PRICE VARIATION DATA TO:\n{pv_path}")

        # select only stocks in portfolio
        if tics_in_portfolio != "all":
            self._df_price_variation = self._df_price_variation[
                self._df_price_variation[self._tic_column].isin(tics_in_portfolio)
            ]
        # apply normalization
        if normalize:
            self._normalize_dataframe(normalize)
        # transform str to datetime
        self._df[self._time_column] = pd.to_datetime(self._df[self._time_column])
        self._df_price_variation[self._time_column] = pd.to_datetime(
            self._df_price_variation[self._time_column]
        )
        # transform numeric variables to float32 (compatibility with pytorch)
        self._df[self._features] = self._df[self._features].astype("float32")
        self._df_price_variation[self._features] = self._df_price_variation[
            self._features
        ].astype("float32")

    def _reset_memory(self):
        """Resets the environment's memory."""
        date_time = self._sorted_times[self._time_index]
        # memorize portfolio value each step
        self._asset_memory = {
            "initial": [self._initial_amount],
            "final": [self._initial_amount],
        }
        # memorize portfolio return and reward each step
        self._portfolio_return_memory = [0]
        self._portfolio_reward_memory = [0]
        # initial action: all money is allocated in cash
        self._actions_memory = [
            np.array([1] + [0] * self.portfolio_size, dtype=np.float32)
        ]
        # memorize portfolio weights at the ending of time step
        self._final_weights = [
            np.array([1] + [0] * self.portfolio_size, dtype=np.float32)
        ]
        # memorize datetimes
        self._date_memory = [date_time]

        # added by miftah
        self._detailed_actions_memory = []
        self._first_episode = True
        tmp = [0] * len(self._tic_list)
        self._prev_weights = np.asarray([1] + tmp)
        self._portfolio_value = self._initial_amount
        self._start_timestamp = dt.now()

        # if self._mode in ["dev", "test"]:
        self._latest_profit = 0
        self._max_profit = -1

    def _standardize_state(self, state):
        """Standardize the state given the observation space. If "return_last_action"
        is set to False, a three-dimensional box is returned. If it's set to True, a
        dictionary is returned. The dictionary follows the standard below::

            {
            "state": Three-dimensional box representing the current state,
            "last_action": One-dimensional box representing the last action
            }
        """
        last_action = self._actions_memory[-1]
        if self._return_last_action:
            return {"state": state, "last_action": last_action}
        else:
            return state

    def _normalize_dataframe(self, normalize):
        """ "Normalizes the environment's dataframe.

        Args:
            normalize: Defines the normalization method applied to the dataframe.
                Possible values are "by_previous_time", "by_fist_time_window_value",
                "by_COLUMN_NAME" (where COLUMN_NAME must be changed to a real column
                name) and a custom function. If None no normalization is done.

        Note:
            If a custom function is used in the normalization, it must have an
            argument representing the environment's dataframe.
        """
        if type(normalize) == str:
            if normalize == "by_fist_time_window_value":
                print(
                    "Normalizing {} by first time window value...".format(
                        self._features
                    )
                )
                self._df = self._temporal_variation_df(self._time_window - 1)
            elif normalize == "by_previous_time":
                print(f"Normalizing {self._features} for data {self._mode} by previous time...")
                pv_path = f"/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/combined_data/75_tic_price_variation_v5_{self._mode}_9_features.csv"
                self._df = None
                if os.path.isfile(pv_path):
                    self._df = pd.read_csv(pv_path)
                    print(f"LOADED NORMALIZED DATA FROM:\n{pv_path}")
                else:
                    self._df = self._temporal_variation_df()
                    self._df.to_csv(pv_path, index=False)
                    print(f"SAVED NORMALIZED DATA TO:\n{pv_path}")
            elif normalize.startswith("by_"):
                normalizer_column = normalize[3:]
                print(f"Normalizing {self._features} by {normalizer_column}")
                for column in self._features:
                    self._df[column] = self._df[column] / self._df[normalizer_column]
            # SAVE NORMALIZATION - MIFTAH
            # pv_path = f"/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/combined_data/75_tic_v3-a_{self._mode}_12_features_n_{normalize}.xlsx"
            # self._df.to_excel(pv_path, sheet_name=self._mode, index=False)
            # print(f"NORMALIZATION DATA IS SAVED TO:\n{pv_path}")
        elif callable(normalize):
            print("Applying custom normalization function...")
            self._df = normalize(self._df)
        else:
            print("No normalization was performed.")

    def _temporal_variation_df(self, periods=1):
        """Calculates the temporal variation dataframe. For each feature, this
        dataframe contains the rate of the current feature's value and the last
        feature's value given a period. It's used to normalize the dataframe.

        Args:
            periods: Periods (in time indexes) to calculate temporal variation.

        Returns:
            Temporal variation dataframe.
        """
        df_temporal_variation = self._df.copy()
        prev_columns = []
        for column in self._features:
            prev_column = f"prev_{column}"
            prev_columns.append(prev_column)
            df_temporal_variation[prev_column] = df_temporal_variation.groupby(
                self._tic_column
            )[column].shift(periods=periods)
            df_temporal_variation[column] = (
                df_temporal_variation[column] / df_temporal_variation[prev_column]
            )
        df_temporal_variation = (
            df_temporal_variation.drop(columns=prev_columns)
            .fillna(1)
            .reset_index(drop=True)
        )
        return df_temporal_variation

    def _seed(self, seed=None):
        """Seeds the sources of randomness of this environment to guarantee
        reproducibility.

        Args:
            seed: Seed value to be applied.

        Returns:
            Seed value applied.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self, env_number=1):
        """Generates an environment compatible with Stable Baselines 3. The
        generated environment is a vectorized version of the current one.

        Returns:
            A tuple with the generated environment and an initial observation.
        """
        e = DummyVecEnv([lambda: self] * env_number)
        obs = e.reset()
        return e, obs
