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
        buying_fee_pct=0,
        selling_fee_pct=0,
        alpha=0.01,
        mode="train", # options: train, dev, test
        use_sortino_ratio=False,
        risk_free_rate=0,
        detailed_actions_file="",
        eval_episode=None, # what training episode is the evaluation (dev, test) currently on
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

        # added by miftah to record detailed actions for each day
        self._detailed_actions_memory = []
        self._start_timestamp = dt.now()
        self._mode = mode
        self.detailed_actions_file = detailed_actions_file
        self._transaction_cost = 0
        self._alpha = alpha # how big is the impact of the transaction cost penalty on reward calculation
        self._prev_weights = None
        self._first_episode = True
        self._use_sortino_ratio = use_sortino_ratio
        self._risk_free_rate = risk_free_rate
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
        self._tic_list = self._df[self._tic_column].unique()
        self.portfolio_size = (
            len(self._tic_list)
            if tics_in_portfolio == "all"
            else len(tics_in_portfolio)
        )
        action_space = 1 + self.portfolio_size

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
        self._terminal = self._time_index >= len(self._sorted_times) - 1

        if self._terminal:
            # d = "/".join(self.detailed_actions_file.split("/")[:-1])
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
                # if self._mode == "train":
                    # tmp = self.detailed_actions_file[:-5]
                    # tmp = f"{tmp}_{episode}.xlsx"
                tmp = f"{d}/result_eiie_{self._mode}.xlsx"
                if self._mode == "train":
                    tmp = f"{d}/result_eiie_{self._mode}_{episode}.xlsx"
                with pd.ExcelWriter(tmp) as writer:
                    # filtered_actions_df.to_excel(writer, sheet_name='detailed_actions', index=False)
                    actions_df.to_excel(writer, sheet_name='detailed_actions', index=False)
                    duration_df.to_excel(writer, sheet_name='duration', index=False)
                # filtered_actions_df.to_csv(self.detailed_actions_file, index=False)
                # print(f"Detailed actions and duration is saved to:\n{self.detailed_actions_file}")
                print(f"Detailed actions and duration is saved to:\n{tmp}")
            else:
                print(f"TERMINAL. SKIP WRITING DETAILED RESULT FOR MODE: {self._mode}, EPISODE: {episode}")

            if self._new_gym_api:
                return self._state, self._reward, self._terminal, False, self._info
            return self._state, self._reward, self._terminal, self._info

        else:
            # transform action to numpy array (if it's a list)
            actions = np.array(actions, dtype=np.float32)
            date_str = self._info["end_time"].strftime("%Y-%m-%d")
            # print(f"ACTIONS FROM EIIE MODEL ON {date_str}: {actions}")
            # print(f"SUM OF THE ACTIONS: {np.sum(actions)}")
            # if np.isnan(actions).any():
            #     raise ValueError(f"FOUND NAN IN MODEL ACTIONS EPISODE {episode}: {actions}")
            # if date_str == "2023-04-06":
            #     raise ValueError(f"STOP")

            # if necessary, normalize weights
            if math.isclose(np.sum(actions), 1, abs_tol=1e-6) and np.min(actions) >= 0:
                weights = actions
            else:
                weights = self._softmax_normalization(actions)
            # print(f"ACTIONS AFTER PROCESSING 1 FROM EIIE MODEL ON {date_str}: {weights}")
            # print(f"SUM OF THE ACTIONS FROM PROCESSING 1: {np.sum(actions)}")

            # POSTPROCESSING WEIGHTS - MIFTAH
            # 1. Round weights to multiplies of 0.05
            weights = np.round(weights * 20) / 20
            # if after this command the sum is zero 0, then we reset the value back
            # print(f"ACTIONS AFTER PROCESSING 2- MULTIPLIES OF 0.05. FROM EIIE MODEL ON {date_str}: {weights}")
            # 2. Set weights < 0.05 to 0
            weights[weights < 0.05] = 0
            # print(f"ACTIONS AFTER PROCESSING 2- SET BELOW 0.05 TO ZERO. FROM EIIE MODEL ON {date_str}: {weights}")
            # Renormalize weights to sum to 1 after zeroing small weights
            # V1
            # weights = weights / np.sum(weights)
            # print(f"ACTIONS AFTER PROCESSING 2- DIVIDE BY SUM OF WEIGHTS. FROM EIIE MODEL ON {date_str}: {weights}")
            # V2
            if np.sum(weights) == 0:
                weights[0] = 1.0
            else:
                weights = weights / np.sum(weights)

            # CALCULATE TODAY'S PORTFOLIO USING WEIGHTS ON PREV DAY - MIFTAH
            # MOVED FROM LINE 448 BELOW
            if self._first_episode:
                tmp = [0] * len(self._info["tics"].tolist())
                self._prev_weights = [1] + tmp
                self._first_episode = False
            # print(episode, self._prev_weights)
            portfolio = self._portfolio_value * (self._prev_weights * self._price_variation)
            self._portfolio_value = np.sum(portfolio)
            self._interim_portfolio_value = np.sum(portfolio)

            # save initial portfolio weights for this time step
            self._actions_memory.append(weights)

            # get last step final weights and portfolio_value
            last_weights = self._final_weights[-1]

            # load next state
            self._time_index += 1
            self._state, self._info = self._get_state_and_info_from_time_index(
                self._time_index
            )

            # if using weights vector modifier, we need to modify weights vector
            if self._comission_fee_model == "wvm":
                delta_weights = weights - last_weights
                delta_assets = delta_weights[1:]  # disconsider
                # calculate fees considering weights modification
                fees = np.sum(np.abs(delta_assets * self._portfolio_value))

                # Store transaction cost for tracking
                self._transaction_cost = fees
                # self._info["transaction_cost"] = fees

                if fees > weights[0] * self._portfolio_value:
                    weights = last_weights
                    # maybe add negative reward
                else:
                    portfolio = weights * self._portfolio_value
                    portfolio[0] -= fees
                    self._portfolio_value = np.sum(portfolio)  # new portfolio value
                    weights = portfolio / self._portfolio_value  # new weights
            elif self._comission_fee_model == "trf":
                last_mu = 1
                mu = 1 - 2 * self._comission_fee_pct + self._comission_fee_pct**2
                while abs(mu - last_mu) > 1e-10:
                    last_mu = mu
                    mu = (
                        1
                        - self._comission_fee_pct * weights[0]
                        - (2 * self._comission_fee_pct - self._comission_fee_pct**2)
                        * np.sum(np.maximum(last_weights[1:] - mu * weights[1:], 0))
                    ) / (1 - self._comission_fee_pct * weights[0])
                self._info["trf_mu"] = mu
                self._portfolio_value = mu * self._portfolio_value
                self._transaction_cost = 1 - mu
            elif self._comission_fee_model == "trf_v2":
                # **TRF_V2 SECTION START - MIFTAH**

                buying_fee_pct = self._buying_fee_pct  # Separate buying fee percentage
                selling_fee_pct = self._selling_fee_pct  # Separate selling fee percentage
                last_mu = 1.0
                # Initialize mu considering both buying and selling fees
                mu = 1.0 - buying_fee_pct - selling_fee_pct
                # mu = 1 - 0.0025 - 0.0025
                # minimum mu = 0.005

                # Iteratively solve for mu to account for transaction costs
                max_iterations = 1000
                iteration = 0
                tolerance = 1e-10

                # Calculate w_i based on price variation
                # ltmp = last_weights * self._price_variation
                ltmp = self._prev_weights * self._price_variation
                w_i = np.zeros_like(ltmp)
                if np.sum(ltmp) > 0:
                    w_i = ltmp / np.sum(ltmp)

                # Define the equation to solve: f(mu) = 0
                # def equation(mu):
                #     # Calculate selling reductions and costs
                #     selling_reduction = np.maximum(w_i[1:] - mu * weights[1:], 0)
                #     selling_cost = selling_fee_pct * np.sum(selling_reduction)
                #     # Calculate buying increases and costs
                #     buying_increase = np.maximum(mu * weights[1:] - w_i[1:], 0)
                #     buying_cost = buying_fee_pct * np.sum(buying_increase)
                #     # Define the equation based on mu
                #     return mu - (1 - buying_fee_pct * weights[0] - selling_cost - buying_cost) / (1 - buying_fee_pct * weights[0])

                # try:
                #     # Initial guess for mu
                #     initial_mu = 1.0 - buying_fee_pct - selling_fee_pct
                #     # Solve for mu using Newton-Raphson method
                #     # Explanation: https://docs.google.com/document/d/11JchD-LxiSccEjjkwzoE7RDqxKQmO0_uMrWAhM2qTG4/edit?usp=sharing
                #     mu = newton(equation, initial_mu, tol=1e-10, maxiter=1000)
                # except RuntimeError:
                #     # Handle the case when the solver did not converge
                #     # Fallback to initial_mu or raise an error
                #     mu = 1.0 - buying_fee_pct - selling_fee_pct
                #     print("Warning. Episode {episode}: mu calculation did not converge. Using initial guess.\n")
                #     # Alternatively, uncomment the next line to raise an error
                #     # raise ValueError("mu calculation did not converge")

                # After finding mu, calculate selling and buying costs - V1
                # selling_reduction = np.maximum(w_i[1:] - mu * weights[1:], 0)
                # selling_cost = selling_fee_pct * np.sum(selling_reduction)
                # buying_increase = np.maximum(mu * weights[1:] - w_i[1:], 0)
                # buying_cost = buying_fee_pct * np.sum(buying_increase)

                # After finding mu, calculate selling and buying costs (MU REMOVED) - V2
                date_str = self._info["end_time"].strftime("%Y-%m-%d")
                selling_reduction = np.maximum(w_i[1:] - weights[1:], 0)
                selling_cost = selling_fee_pct * np.sum(selling_reduction)
                buying_increase = np.maximum(weights[1:] - w_i[1:], 0)
                buying_cost = buying_fee_pct * np.sum(buying_increase)
                # print(f"DATE: {date_str}")
                # print(f"SELLING REDUCTION: {selling_reduction}")
                # print(f"BUYING INCREASE: {buying_increase}")
                # print(f"W_I: {w_i[1:]}")
                # print(f"WEIGHTS: {weights[1:]}\n")
                # if date_str == "2023-09-25":
                #     raise ValueError(f"STOP")

                # Assign separate transaction costs
                self._selling_cost = selling_cost
                self._buying_cost = buying_cost

                # Calculate total transaction cost
                self._transaction_cost = self._selling_cost + self._buying_cost

                # Update portfolio value
                self._info["trf_mu"] = mu
                # self._portfolio_value = mu * self._portfolio_value
                # self._portfolio_value = self._portfolio_value - self._transaction_cost
                # print(f"SELF._PORTFOLIO_VALUE BEFORE: {self._portfolio_value}")
                self._portfolio_value = self._portfolio_value * (1 - self._transaction_cost)
                # print(f"SELF._PORTFOLIO_VALUE AFTER: {self._portfolio_value}")
                # self._portfolio_value = self._portfolio_value_prev * (1 - self._transaction_cost)

                # **TRF_V2 SECTION END**

            # save initial portfolio value of this time step
            self._asset_memory["initial"].append(self._portfolio_value)

            # SET NAN WEIGHT TO 0 - MIFTAH
            # weights[np.isnan(weights)] = 0
            # if np.sum(weights) > 1:
            #     weights = self._softmax_normalization(weights)

            # time passes and time variation changes the portfolio distribution
            # portfolio = self._portfolio_value * (weights * self._price_variation)
            # CALCULATE TODAY'S PORTFOLIO USING WEIGHTS ON PREV DAY - MIFTAH
            # if self._first_episode:
            #     tmp = [0] * len(self._info["tics"].tolist())
            #     self._prev_weights = [1] + tmp
            #     self._first_episode = False
            # # print(episode, self._prev_weights)
            # portfolio = self._portfolio_value * (self._prev_weights * self._price_variation)
            self._prev_weights = weights

            date_str = self._info["end_time"].strftime("%Y-%m-%d")
            daily_log = {
                "date": date_str,
                "portfolio": self._portfolio_value,
                "interim_portfolio": self._interim_portfolio_value,
                "buying_cost": self._buying_cost,
                "selling_cost": self._selling_cost,
                "transaction_cost": self._transaction_cost,
            }
            # print(date_str, self._portfolio_value)

            # CALCULATE WEIGHTS PERCENT - MIFTAH
            weights_percent = weights * 100
            weights_percent[weights_percent < 0.01] = 0  # Set small weights to zero
            weight_details = ["CASH"] + self._info["tics"].tolist() # add column for cash allocation
            for i, ticker in enumerate(weight_details):
                daily_log[ticker] = weights_percent[i]
            for i, ticker in enumerate(weight_details):
                daily_log[f"{ticker}_v"] = self._price_variation[i]

            # calculate new portfolio value and weights
            # self._portfolio_value = np.sum(portfolio) # MOVED TO LINE 358
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
            # print(f"PORTFOLIO REWARD: {portfolio_reward}")
            if portfolio_reward > 0:
                portfolio_reward = np.log(portfolio_reward - (self._alpha * self._transaction_cost))

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

        self._state, self._info = self._get_state_and_info_from_time_index(
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
                "end_time_index": Index of end time of current time window,
                "data": Data related to the current time window,
                "price_variation": Price variation of current time step
                }
        """
        # returns state in form (channels, tics, timesteps)
        end_time = self._sorted_times[time_index]
        start_time = self._sorted_times[time_index - (self._time_window - 1)]

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
        return self._standardize_state(state), info

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
        self._df_price_variation = self._temporal_variation_df()

        # SAVE PRICE VARIATION - MIFTAH
        # pv_path = f"/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/combined_data/75_tic_price_variation_v3-a_{self._mode}_12_features.csv"
        # self._df_price_variation.to_csv(pv_path, index=False)

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
        self._prev_weights = None
        self._portfolio_value = self._initial_amount

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
                self._df = self._temporal_variation_df()
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
