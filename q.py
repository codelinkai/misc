import gym
from gym import spaces
import numpy as np
import random
from stable_baselines3 import DQN
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import DQN

class PortfolioQEnv(gym.Env):
    """
    Custom Portfolio Optimization Environment for Q-Learning.
    """

    def __init__(self, df):
        super(PortfolioQEnv, self).__init__()
        self.df = df
        self.unique_dates = sorted(df["date"].unique())
        self.current_step = 0
        self.num_stocks = len(df["ticker"].unique())

        # Define Observation Space
        num_features = df.shape[1] - 2  # Excluding 'date' and 'ticker'
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,))

        # Convert MultiDiscrete to Discrete (Fix)
        self.action_space = spaces.Discrete(3 ** self.num_stocks)  # Reduce MultiDiscrete to a single discrete number

        # Track portfolio
        self.portfolio_allocations = np.full(self.num_stocks, 1 / self.num_stocks)
        self.portfolio_returns = []
        self.portfolio_value = [1.0]  # Start with $1

    def decode_action(self, action):
        """
        Convert a single discrete action into MultiDiscrete actions.
        """
        action_list = []
        base = 3  # Three actions: [-1, 0, +1]
        for _ in range(self.num_stocks):
            action_list.append(action % base - 1)  # Convert to (-1, 0, 1)
            action //= base
        return np.array(action_list)

    def step(self, action):
        """
        Execute one time step.
        """
        action = self.decode_action(action)  # Convert to multi-action

        # Adjust portfolio allocation
        self.portfolio_allocations += action * 0.05  # Adjust weights by ±5%
        self.portfolio_allocations = np.maximum(self.portfolio_allocations, 0)
        self.portfolio_allocations /= np.sum(self.portfolio_allocations)  # Normalize to sum 1.0

        # Compute reward
        today_returns = self.df.pivot(index="date", columns="ticker", values="daily_price").pct_change().iloc[self.current_step]
        portfolio_return = np.dot(self.portfolio_allocations, today_returns.fillna(0).values)
        
        # Compute Sharpe Ratio and Max Drawdown
        if self.current_step >= 10:
            past_returns = self.portfolio_returns[-10:]
            avg_return = np.mean(past_returns)
            volatility = np.std(past_returns) + 1e-8  # Avoid division by zero
            sharpe_ratio = avg_return / volatility
        else:
            sharpe_ratio = 0

        max_drawdown = np.min(self.portfolio_value[: self.current_step] - np.maximum.accumulate(self.portfolio_value[: self.current_step]))

        # Reward = Sharpe Ratio - 0.1 * Max Drawdown
        reward = sharpe_ratio - 0.1 * max_drawdown

        # Store portfolio history
        self.portfolio_returns.append(portfolio_return)
        self.portfolio_value.append(self.portfolio_value[-1] * (1 + portfolio_return))

        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.unique_dates) - 1

        return self._get_observation(), reward, done, {}

    def reset(self):
        """
        Reset the environment.
        """
        self.current_step = 0
        self.portfolio_allocations = np.full(self.num_stocks, 1 / self.num_stocks)
        self.portfolio_returns = []
        self.portfolio_value = [1.0]
        return self._get_observation()
class PortfolioQEnv(gym.Env):
    """
    Custom Portfolio Optimization Environment for Q-Learning.
    """

    def __init__(self, df):
        super(PortfolioQEnv, self).__init__()
        self.df = df
        self.unique_dates = sorted(df["date"].unique())
        self.current_step = 0
        self.num_stocks = len(df["ticker"].unique())

        # Define Observation Space: RSI, MACD, Bollinger Bands, and sector info
        num_features = df.shape[1] - 2  # Excluding 'date' and 'ticker'
        self.observation_space = spaces.Discrete(num_features)  # Q-learning uses discrete states

        # Define Action Space: Increase, Decrease, Hold for each stock
        self.actions = [-1, 0, 1]  # -1 = decrease, 0 = hold, 1 = increase
        self.action_space = spaces.MultiDiscrete([3] * self.num_stocks)  # One action per stock

        # Track portfolio history
        self.portfolio_allocations = np.full(self.num_stocks, 1 / self.num_stocks)  # Equal weights
        self.portfolio_returns = []
        self.portfolio_value = [1.0]  # Start with $1 (normalized)

    def _get_observation(self):
        """
        Get the current observation (excluding 'date' and 'ticker').
        """
        return self.df.iloc[self.current_step].drop(["date", "ticker"]).values

    def step(self, action):
        """
        Execute one time step.
        """
        action = np.array(action) - 1  # Convert action values (-1, 0, 1)
        
        # Update portfolio allocation
        self.portfolio_allocations += action * 0.05  # Adjust weights by ±5%
        self.portfolio_allocations = np.maximum(self.portfolio_allocations, 0)  # Ensure non-negative
        self.portfolio_allocations /= np.sum(self.portfolio_allocations)  # Normalize to sum 1.0

        # Get today's returns (NO FUTURE DATA USED)
        today_returns = self.df.pivot(index="date", columns="ticker", values="daily_price").pct_change().iloc[self.current_step]
        portfolio_return = np.dot(self.portfolio_allocations, today_returns.fillna(0).values)

        # Compute Sharpe Ratio using past returns (NO FUTURE DATA USED)
        if self.current_step >= 10:
            past_returns = self.portfolio_returns[-10:]  # Last 10 days
            avg_return = np.mean(past_returns)
            volatility = np.std(past_returns) + 1e-8  # Avoid division by zero
            sharpe_ratio = avg_return / volatility
        else:
            sharpe_ratio = 0  # Not enough history

        # Compute Max Drawdown using past values (NO FUTURE DATA USED)
        max_drawdown = np.min(self.portfolio_value[: self.current_step] - np.maximum.accumulate(self.portfolio_value[: self.current_step]))

        # Reward function (Sharpe Ratio - 0.1 * Max Drawdown)
        reward = sharpe_ratio - 0.1 * max_drawdown

        # Store portfolio history
        self.portfolio_returns.append(portfolio_return)
        self.portfolio_value.append(self.portfolio_value[-1] * (1 + portfolio_return))

        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.unique_dates) - 1

        return self._get_observation(), reward, done, {}

    def reset(self):
        """
        Reset the environment.
        """
        self.current_step = 0
        self.portfolio_allocations = np.full(self.num_stocks, 1 / self.num_stocks)  # Reset allocations
        self.portfolio_returns = []
        self.portfolio_value = [1.0]
        return self._get_observation()



# Initialize environment
env = PortfolioQEnv(df)

# Train the RL agent using DQN
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Save trained model
model.save("dqn_portfolio")

# Load the trained model
model = DQN.load("dqn_portfolio")

# Run the agent on test data
obs = env.reset()
done = False
portfolio_values = []

while not done:
    action, _states = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    portfolio_values.append(env.portfolio_value[-1])

# Plot portfolio growth
import matplotlib.pyplot as plt
plt.plot(portfolio_values)
plt.title("Portfolio Value Over Time")
plt.xlabel("Time Steps")
plt.ylabel("Portfolio Value ($)")
plt.show()
