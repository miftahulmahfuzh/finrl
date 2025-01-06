from __future__ import annotations

import copy

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .architectures import EIIE
from .utils import apply_portfolio_noise
from .utils import PVM
from .utils import ReplayBuffer
from .utils import RLDataset

# below lines are added by miftah
import os
import json
import matplotlib.pyplot as plt


class PolicyGradient:
    """Class implementing policy gradient algorithm to train portfolio
    optimization agents.

    Note:
        During testing, the agent is optimized through online learning.
        The parameters of the policy is updated repeatedly after a constant
        period of time. To disable it, set learning rate to 0.

    Attributes:
        train_env: Environment used to train the agent
        train_policy: Policy used in training.
        test_env: Environment used to test the agent.
        test_policy: Policy after test online learning.
    """

    def __init__(
        self,
        env,
        policy=EIIE,
        policy_kwargs=None,
        validation_env=None,
        batch_size=100,
        lr=1e-3,
        action_noise=0,
        optimizer=AdamW,
        device="cpu",
        # below lines are added by miftah
        checkpoint_dir="",
        policy_str="",
        dev_env=None,
        test_env=None,
        use_reward_in_loss=False
    ):
        """Initializes Policy Gradient for portfolio optimization.

        Args:
          env: Training Environment.
          policy: Policy architecture to be used.
          policy_kwargs: Arguments to be used in the policy network.
          validation_env: Validation environment.
          batch_size: Batch size to train neural network.
          lr: policy Neural network learning rate.
          action_noise: Noise parameter (between 0 and 1) to be applied
            during training.
          optimizer: Optimizer of neural network.
          device: Device where neural network is run.
        """
        self.policy = policy
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.validation_env = validation_env
        self.batch_size = batch_size
        self.lr = lr
        self.action_noise = action_noise
        # print(f"SELF.ACTION_NOISE: {action_noise}")
        self.optimizer = optimizer
        self.device = device
        self._setup_train(env, self.policy, self.batch_size, self.lr, self.optimizer)

        # below lines are added by miftah
        self._checkpoint_dir = checkpoint_dir
        # print(f"SELF._CHECKPOINT_DIR: {checkpoint_dir}")
        # main_dir = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/data_1993_to_2024/train"
        # d = f"{main_dir}/v2_result/EIIE"
        # self._checkpoint_dir = f"{d}/checkpoint"
        self._asset_history = {
            "train": [],
            "dev": [],
            "test": [],
            "episodes": []
        }

        self._policy_str = policy_str
        self._dev_env = dev_env
        self._test_env = test_env
        self._use_reward_in_loss = use_reward_in_loss
        # print(f"USE REWARD IN LOSS: {self._use_reward_in_loss}")

    def _setup_train(self, env, policy, batch_size, lr, optimizer):
        """Initializes algorithm before training.

        Args:
          env: environment.
          policy: Policy architecture to be used.
          batch_size: Batch size to train neural network.
          lr: Policy neural network learning rate.
          optimizer: Optimizer of neural network.
        """
        # environment
        self.train_env = env

        # neural networks
        self.train_policy = policy(**self.policy_kwargs).to(self.device)
        self.train_optimizer = optimizer(self.train_policy.parameters(), lr=lr)

        # replay buffer and portfolio vector memory
        self.train_batch_size = batch_size
        self.train_buffer = ReplayBuffer(capacity=batch_size)
        self.train_pvm = PVM(self.train_env.episode_length, env.portfolio_size)

        # dataset and dataloader
        dataset = RLDataset(self.train_buffer)
        self.train_dataloader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )

    def train(self, episodes=100):
        """Training sequence.

        Args:
            episodes: Number of episodes to simulate.
        """
        # print(f"IN algorithms.py-PolicyGradient.train(). TOTAL EPISODES: {episodes}")
        for episode in tqdm(range(1, episodes + 1)):
            # print(f"\nEPISODE: {episode}")
            obs = self.train_env.reset()  # observation
            self.train_pvm.reset()  # reset portfolio vector memory
            done = False

            while not done:
                # define last_action and action and update portfolio vector memory
                last_action = self.train_pvm.retrieve()
                obs_batch = np.expand_dims(obs, axis=0)
                last_action_batch = np.expand_dims(last_action, axis=0)
                action = apply_portfolio_noise(
                    self.train_policy(obs_batch, last_action_batch), self.action_noise
                )
                self.train_pvm.add(action)
                # run simulation step

                next_obs, reward, done, info = self.train_env.step(action, episode)

                # add experience to replay buffer
                # exp = (obs, last_action, info["price_variation"], info["trf_mu"])
                # added reward to calculate during gradient_ascent - miftah
                exp = (obs, last_action, info["price_variation"], info["trf_mu"], reward)
                self.train_buffer.append(exp)

                # update policy networks
                if len(self.train_buffer) == self.train_batch_size:
                    self._gradient_ascent()

                obs = next_obs

            # SAVE MODEL AT THE END OF EPISODE - MIFTAH
            self._save_model(episode)

            # EVALUATE AT THE END OF EPISODE - MIFTAH
            policy = self.train_policy
            online_training_period = 10
            learning_rate = self.lr
            optimizer = self.optimizer

            self._dev_env._reset_memory()
            self._dev_env._eval_episode = episode
            self.test(self._dev_env, policy, online_training_period, learning_rate, optimizer)

            self._test_env._reset_memory()
            self._test_env._eval_episode = episode
            self.test(self._test_env, policy, online_training_period, learning_rate, optimizer)

            asset_episode = {
                "train": int(self.train_env._asset_memory["final"][-1]),
                "dev": int(self._dev_env._asset_memory["final"][-1]),
                "test": int(self._test_env._asset_memory["final"][-1])
            }
            d = f"{self._checkpoint_dir}-{episode}"
            os.makedirs(d, exist_ok=True)
            outf = f"{d}/asset_episode.json"
            x = json.dumps(asset_episode, indent=3)
            print(f"ASSET PATH: {outf}")
            with open(outf, "w+") as f:
                f.write(x)

            # self._asset_memory["final"].append(self._portfolio_value)

            # gradient ascent with episode remaining buffer data
            self._gradient_ascent()
            self.train_env._reset_memory()

            # validation step
            if self.validation_env:
                self.test(self.validation_env)

            # -----------------------------------------------------------------
            # PLOT 3 graphs (train, dev, test) and save
            # -----------------------------------------------------------------
            # 1. Accumulate each mode's final asset in self._asset_history
            self._asset_history["train"].append(asset_episode["train"])
            self._asset_history["dev"].append(asset_episode["dev"])
            self._asset_history["test"].append(asset_episode["test"])
            self._asset_history["episodes"].append(episode)

            # 2. For each mode (train/dev/test), generate and save a plot of
            #    Rate of Return (%) vs Episode.

            for mode in ["train", "dev", "test"]:
                # Calculate rate of return (%) for each episode in history
                # final_asset / initial_amount * 100
                returns = [
                    (asset / self.train_env._initial_amount) * 100.0
                    for asset in self._asset_history[mode]
                ]

                # Create a new figure per mode
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(self._asset_history["episodes"], returns, marker='o', label=mode.upper())

                ax.set_title(f"Rate of Return for {mode.upper()}")
                ax.set_xlabel("Episode")
                ax.set_ylabel("Return (%)")
                ax.grid(True)
                ax.legend(loc="best")

                # Save the figure in the checkpoint directory for this episode
                dg = self._checkpoint_dir.split("/")[:-1]
                dg = "/".join(dg)
                outpath = f"{dg}/{mode}.png"
                plt.savefig(outpath)
                plt.close(fig)  # close to free memory

            # End of PLOT block

    def _save_model(self, episode):
        d = f"{self._checkpoint_dir}-{episode}"
        os.makedirs(d, exist_ok=True)
        outf = f"{d}/policy_{self._policy_str}_{episode}.pt"
        torch.save(self.train_policy.state_dict(), outf)
        print(f"{self._policy_str} Model Episode {episode} is saved to:\n{outf}")

    def _setup_test(self, env, policy, batch_size, lr, optimizer):
        """Initializes algorithm before testing.

        Args:
          env: Environment.
          policy: Policy architecture to be used.
          batch_size: batch size to train neural network.
          lr: policy neural network learning rate.
          optimizer: Optimizer of neural network.
        """
        # environment
        self.test_env = env

        # process None arguments
        policy = self.train_policy if policy is None else policy
        lr = self.lr if lr is None else lr
        optimizer = self.optimizer if optimizer is None else optimizer

        # neural networks
        # define policy
        self.test_policy = copy.deepcopy(policy).to(self.device)
        self.test_optimizer = optimizer(self.test_policy.parameters(), lr=lr)

        # replay buffer and portfolio vector memory
        self.test_buffer = ReplayBuffer(capacity=batch_size)
        # print(f"SELF TEST ENV EPISODE LENGTH {self.test_env.episode_length}")
        # print(f"ENV PORTFOLIO SIZE {env.portfolio_size}")
        self.test_pvm = PVM(self.test_env.episode_length, env.portfolio_size)

        # dataset and dataloader
        dataset = RLDataset(self.test_buffer)
        self.test_dataloader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )

    def test(
        self, env, policy=None, online_training_period=10, lr=None, optimizer=None
    ):
        """Tests the policy with online learning.

        Args:
          env: Environment to be used in testing.
          policy: Policy architecture to be used. If None, it will use the training
            architecture.
          online_training_period: Period in which an online training will occur. To
            disable online learning, use a very big value.
          batch_size: Batch size to train neural network. If None, it will use the
            training batch size.
          lr: Policy neural network learning rate. If None, it will use the training
            learning rate
          optimizer: Optimizer of neural network. If None, it will use the training
            optimizer

        Note:
            To disable online learning, set learning rate to 0 or a very big online
            training period.
        """
        self._setup_test(env, policy, online_training_period, lr, optimizer)

        obs = self.test_env.reset()  # observation
        self.test_pvm.reset()  # reset portfolio vector memory
        done = False
        steps = 0

        while not done:
            steps += 1
            # define last_action and action and update portfolio vector memory
            last_action = self.test_pvm.retrieve()
            # print(f"LAST ACTION: {last_action}")
            obs_batch = np.expand_dims(obs, axis=0)
            last_action_batch = np.expand_dims(last_action, axis=0)
            action = self.test_policy(obs_batch, last_action_batch)
            # print(f"ACTION: {action}")
            self.test_pvm.add(action)

            # run simulation step
            next_obs, reward, done, info = self.test_env.step(action)
            # print(f"DONE: {done}")
            # print(f"INFO: {info['price_variation'].shape}")

            # add experience to replay buffer
            # exp = (obs, last_action, info["price_variation"], info["trf_mu"])
            # added reward to calculate during gradient_ascent - miftah
            exp = (obs, last_action, info["price_variation"], info["trf_mu"], reward)
            self.test_buffer.append(exp)

            # update policy networks
            if steps % online_training_period == 0:
                self._gradient_ascent(test=True)

            obs = next_obs

    def _gradient_ascent(self, test=False):
        """Performs the gradient ascent step in the policy gradient algorithm.

        Args:
            test: If true, it uses the test dataloader and policy.
        """
        # get batch data from dataloader
        # print(f"TEST: {test}")
        obs, last_actions, price_variations, trf_mu, reward = (
            next(iter(self.test_dataloader))
            if test
            else next(iter(self.train_dataloader))
        )
        obs = obs.to(self.device)
        last_actions = last_actions.to(self.device)
        price_variations = price_variations.to(self.device)
        trf_mu = trf_mu.unsqueeze(1).to(self.device)
        # print(f"REWARD BEFORE UNSQUEEZE: {reward}\n")

        # define policy loss (negative for gradient ascent)
        mu = (
            self.test_policy.mu(obs, last_actions)
            if test
            else self.train_policy.mu(obs, last_actions)
        )
        policy_loss_tmp = None
        if self._use_reward_in_loss:
            reward = reward.unsqueeze(1).to(self.device)
            policy_loss_tmp = torch.log(torch.sum(mu * price_variations * trf_mu, dim=1)) * reward
        else:
            policy_loss_tmp = torch.log(torch.sum(mu * price_variations * trf_mu, dim=1))
        policy_loss = -torch.mean(policy_loss_tmp)
        # policy_loss = -torch.mean(
        #     torch.log(torch.sum(mu * price_variations * trf_mu, dim=1)) * reward
        # )
        # print(f"REWARD: {reward}. POLICY_LOSS: {policy_loss}")
        # TRY TO ADD REWARD IN POLICY_LOSS
        # CHANGED BY MIFTAH - REMOVED trf_mu
        # policy_loss = -torch.mean(
        #     torch.log(torch.sum(mu * price_variations, dim=1))
        # )

        # update policy network
        if test:
            self.test_policy.zero_grad()
            policy_loss.backward()
            self.test_optimizer.step()
        else:
            self.train_policy.zero_grad()
            policy_loss.backward()
            self.train_optimizer.step()
